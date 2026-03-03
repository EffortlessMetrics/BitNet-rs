#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]
#![cfg(feature = "cpu")]

//! Metal memory management tests for Apple Silicon.
//!
//! Validates allocation strategies, pool management, lifecycle tracking,
//! pressure handling, and Apple Silicon unified memory patterns using
//! pure mock types — no GPU hardware or Metal/wgpu crates required.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

// ══════════════════════════════════════════════════════════════════════
// Mock types for Metal memory management simulation
// ══════════════════════════════════════════════════════════════════════

/// Metal storage mode (mirrors `MTLStorageMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StorageMode {
    /// GPU-only; fastest for pure compute.
    Private,
    /// CPU + GPU coherent; ideal for Apple Silicon UMA.
    Shared,
    /// CPU-managed with explicit sync points.
    Managed,
}

/// Metal CPU cache mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuCacheMode {
    DefaultCache,
    WriteCombined,
}

/// Unique buffer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferId(u64);

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_buffer_id() -> BufferId {
    BufferId(NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed))
}

/// Simulated Metal buffer.
#[derive(Debug, Clone)]
struct MockBuffer {
    id: BufferId,
    requested_size: usize,
    allocated_size: usize,
    storage_mode: StorageMode,
    cache_mode: CpuCacheMode,
    label: String,
    ref_count: u32,
    contents: Vec<u8>,
}

impl MockBuffer {
    fn new(size: usize, storage: StorageMode, label: &str) -> Self {
        let aligned = align_up(size, 16);
        Self {
            id: next_buffer_id(),
            requested_size: size,
            allocated_size: aligned,
            storage_mode: storage,
            cache_mode: CpuCacheMode::DefaultCache,
            label: label.to_string(),
            ref_count: 1,
            contents: vec![0u8; aligned],
        }
    }
}

/// Texture dimension kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureDimension {
    D2,
    D3,
}

/// Pixel format (subset relevant to inference).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PixelFormat {
    R32Float,
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
}

impl PixelFormat {
    fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::R32Float => 4,
            PixelFormat::Rgba8Unorm => 4,
            PixelFormat::Rgba16Float => 8,
            PixelFormat::Rgba32Float => 16,
        }
    }
}

/// Simulated texture descriptor.
#[derive(Debug, Clone)]
struct TextureDescriptor {
    width: u32,
    height: u32,
    depth: u32,
    dimension: TextureDimension,
    format: PixelFormat,
    mipmap_levels: u32,
}

impl TextureDescriptor {
    fn base_size_bytes(&self) -> usize {
        self.width as usize
            * self.height as usize
            * self.depth as usize
            * self.format.bytes_per_pixel()
    }

    fn total_size_with_mipmaps(&self) -> usize {
        let mut total = 0usize;
        let mut w = self.width as usize;
        let mut h = self.height as usize;
        for _ in 0..self.mipmap_levels {
            total += w * h * self.depth as usize * self.format.bytes_per_pixel();
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }
        total
    }
}

/// Eviction policy for memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvictionPolicy {
    Lru,
    Priority,
}

/// Priority level for buffer retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BufferPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn align_up(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (size + alignment - 1) & !(alignment - 1)
}

fn next_power_of_2(v: usize) -> usize {
    v.next_power_of_two()
}

// ══════════════════════════════════════════════════════════════════════
// 1. Buffer Allocation Strategy
// ══════════════════════════════════════════════════════════════════════

/// Allocator that rounds sizes to power-of-2 buckets and reuses freed buffers.
struct BucketAllocator {
    free_lists: BTreeMap<usize, Vec<MockBuffer>>,
    allocated: HashMap<BufferId, MockBuffer>,
    total_allocated: usize,
}

impl BucketAllocator {
    fn new() -> Self {
        Self { free_lists: BTreeMap::new(), allocated: HashMap::new(), total_allocated: 0 }
    }

    fn bucket_size(requested: usize) -> usize {
        let min = 256;
        next_power_of_2(requested.max(min))
    }

    fn allocate(&mut self, size: usize, storage: StorageMode, label: &str) -> BufferId {
        let bucket = Self::bucket_size(size);
        if let Some(list) = self.free_lists.get_mut(&bucket) {
            if let Some(mut buf) = list.pop() {
                buf.requested_size = size;
                buf.label = label.to_string();
                buf.ref_count = 1;
                let id = buf.id;
                self.allocated.insert(id, buf);
                return id;
            }
        }
        let mut buf = MockBuffer::new(size, storage, label);
        buf.allocated_size = bucket;
        buf.contents = vec![0u8; bucket];
        let id = buf.id;
        self.total_allocated += bucket;
        self.allocated.insert(id, buf);
        id
    }

    fn release(&mut self, id: BufferId) -> bool {
        if let Some(buf) = self.allocated.remove(&id) {
            let bucket = buf.allocated_size;
            self.free_lists.entry(bucket).or_default().push(buf);
            true
        } else {
            false
        }
    }

    fn active_count(&self) -> usize {
        self.allocated.len()
    }

    fn free_count(&self) -> usize {
        self.free_lists.values().map(|v| v.len()).sum()
    }
}

#[test]
fn alloc_power_of_2_rounding() {
    assert_eq!(BucketAllocator::bucket_size(100), 256);
    assert_eq!(BucketAllocator::bucket_size(257), 512);
    assert_eq!(BucketAllocator::bucket_size(1023), 1024);
    assert_eq!(BucketAllocator::bucket_size(1024), 1024);
    assert_eq!(BucketAllocator::bucket_size(1025), 2048);
}

#[test]
fn alloc_minimum_bucket_size() {
    assert_eq!(BucketAllocator::bucket_size(1), 256);
    assert_eq!(BucketAllocator::bucket_size(0), 256);
    assert_eq!(BucketAllocator::bucket_size(128), 256);
    assert_eq!(BucketAllocator::bucket_size(256), 256);
}

#[test]
fn alloc_alignment_16_byte() {
    let buf = MockBuffer::new(17, StorageMode::Shared, "test");
    assert_eq!(buf.allocated_size % 16, 0);
    assert!(buf.allocated_size >= 17);
    assert_eq!(buf.allocated_size, 32);
}

#[test]
fn alloc_alignment_256_byte() {
    for &sz in &[1, 100, 255, 256, 257, 512] {
        let aligned = align_up(sz, 256);
        assert_eq!(aligned % 256, 0);
        assert!(aligned >= sz);
    }
}

#[test]
fn alloc_alignment_4096_page() {
    for &sz in &[1, 4095, 4096, 4097, 8192] {
        let aligned = align_up(sz, 4096);
        assert_eq!(aligned % 4096, 0);
        assert!(aligned >= sz);
    }
}

#[test]
fn alloc_size_limits() {
    let max_buffer: usize = 256 * 1024 * 1024 * 1024; // 256 GiB
    assert!(BucketAllocator::bucket_size(1024) <= max_buffer);
    let large = 1usize << 30; // 1 GiB
    let bucket = BucketAllocator::bucket_size(large);
    assert!(bucket.is_power_of_two());
    assert!(bucket >= large);
}

#[test]
fn alloc_reuse_from_pool() {
    let mut alloc = BucketAllocator::new();
    let id1 = alloc.allocate(100, StorageMode::Shared, "a");
    alloc.release(id1);
    assert_eq!(alloc.free_count(), 1);

    let id2 = alloc.allocate(80, StorageMode::Shared, "b");
    // Should reuse freed buffer (same 256-byte bucket)
    assert_eq!(alloc.free_count(), 0);
    assert_eq!(alloc.active_count(), 1);
    let _ = id2;
}

#[test]
fn alloc_no_reuse_different_bucket() {
    let mut alloc = BucketAllocator::new();
    let id1 = alloc.allocate(100, StorageMode::Shared, "small");
    alloc.release(id1);

    // 2048-byte bucket won't match the 256-byte freed buffer
    let _id2 = alloc.allocate(1500, StorageMode::Shared, "large");
    assert_eq!(alloc.free_count(), 1); // small buffer still free
    assert_eq!(alloc.active_count(), 1);
}

// ══════════════════════════════════════════════════════════════════════
// 2. Memory Pool Management
// ══════════════════════════════════════════════════════════════════════

struct MemoryPool {
    name: String,
    capacity: usize,
    used: usize,
    buffers: Vec<MockBuffer>,
    fragment_count: usize,
    compaction_threshold: f64,
}

impl MemoryPool {
    fn new(name: &str, capacity: usize) -> Self {
        Self {
            name: name.to_string(),
            capacity,
            used: 0,
            buffers: Vec::new(),
            fragment_count: 0,
            compaction_threshold: 0.3,
        }
    }

    fn allocate(&mut self, size: usize, label: &str) -> Option<BufferId> {
        let aligned = align_up(size, 256);
        if self.used + aligned > self.capacity {
            return None;
        }
        let buf = MockBuffer::new(aligned, StorageMode::Shared, label);
        let id = buf.id;
        self.used += aligned;
        self.buffers.push(buf);
        Some(id)
    }

    fn release(&mut self, id: BufferId) -> bool {
        if let Some(pos) = self.buffers.iter().position(|b| b.id == id) {
            let buf = self.buffers.remove(pos);
            self.used -= buf.allocated_size;
            self.fragment_count += 1;
            true
        } else {
            false
        }
    }

    fn grow(&mut self, additional: usize) {
        self.capacity += additional;
    }

    fn shrink(&mut self, amount: usize) -> bool {
        let free = self.capacity - self.used;
        if amount <= free {
            self.capacity -= amount;
            true
        } else {
            false
        }
    }

    fn fragmentation_ratio(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        let free = self.capacity - self.used;
        if free == 0 {
            return 0.0;
        }
        // fragmentation = number of gaps relative to free space
        self.fragment_count as f64 / (free as f64 / 256.0).max(1.0)
    }

    fn needs_compaction(&self) -> bool {
        self.fragmentation_ratio() > self.compaction_threshold
    }

    fn compact(&mut self) {
        self.fragment_count = 0;
    }

    fn utilisation(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.used as f64 / self.capacity as f64
    }
}

#[test]
fn pool_creation_and_capacity() {
    let pool = MemoryPool::new("compute", 1024 * 1024);
    assert_eq!(pool.capacity, 1024 * 1024);
    assert_eq!(pool.used, 0);
    assert_eq!(pool.buffers.len(), 0);
}

#[test]
fn pool_allocate_and_track() {
    let mut pool = MemoryPool::new("compute", 4096);
    let id = pool.allocate(256, "weights").unwrap();
    assert_eq!(pool.buffers.len(), 1);
    assert_eq!(pool.used, 256);
    let _ = id;
}

#[test]
fn pool_exhaustion() {
    let mut pool = MemoryPool::new("small", 512);
    let _id1 = pool.allocate(256, "a").unwrap();
    let _id2 = pool.allocate(256, "b").unwrap();
    assert!(pool.allocate(256, "c").is_none()); // pool full
}

#[test]
fn pool_grow_allows_more() {
    let mut pool = MemoryPool::new("growable", 512);
    let _id = pool.allocate(512, "fill").unwrap();
    assert!(pool.allocate(256, "overflow").is_none());
    pool.grow(512);
    assert!(pool.allocate(256, "after_grow").is_some());
}

#[test]
fn pool_shrink_respects_used() {
    let mut pool = MemoryPool::new("shrinkable", 2048);
    let _id = pool.allocate(1024, "data").unwrap();
    assert!(!pool.shrink(1536)); // can't shrink below used
    assert!(pool.shrink(512)); // free space is 1024, shrink by 512 ok
    assert_eq!(pool.capacity, 1536);
}

#[test]
fn pool_fragmentation_tracking() {
    let mut pool = MemoryPool::new("frag", 4096);
    let id1 = pool.allocate(256, "a").unwrap();
    let _id2 = pool.allocate(256, "b").unwrap();
    let id3 = pool.allocate(256, "c").unwrap();
    pool.release(id1);
    pool.release(id3);
    assert!(pool.fragment_count >= 2);
    assert!(pool.fragmentation_ratio() > 0.0);
}

#[test]
fn pool_compaction_trigger() {
    let mut pool = MemoryPool::new("compact", 4096);
    // Allocate then free many to drive up fragmentation
    let ids: Vec<_> = (0..8).map(|i| pool.allocate(256, &format!("buf{i}")).unwrap()).collect();
    for id in ids.into_iter().step_by(2) {
        pool.release(id);
    }
    if pool.needs_compaction() {
        pool.compact();
        assert_eq!(pool.fragment_count, 0);
    }
}

#[test]
fn pool_utilisation_ratio() {
    let mut pool = MemoryPool::new("util", 1024);
    let _ = pool.allocate(512, "half").unwrap();
    let u = pool.utilisation();
    assert!((u - 0.5).abs() < 0.01);
}

// ══════════════════════════════════════════════════════════════════════
// 3. Shared Memory Allocation
// ══════════════════════════════════════════════════════════════════════

struct StorageModeSelector;

impl StorageModeSelector {
    /// Pick the best storage mode for a given access pattern.
    fn select(cpu_read: bool, cpu_write: bool, gpu_write: bool) -> StorageMode {
        match (cpu_read, cpu_write, gpu_write) {
            (false, false, _) => StorageMode::Private,
            (true, true, _) => StorageMode::Shared,
            (true, false, true) => StorageMode::Managed,
            (_, true, false) => StorageMode::Shared,
            _ => StorageMode::Shared,
        }
    }
}

/// Simulates CPU↔GPU coherence with explicit sync points.
struct CoherenceTracker {
    cpu_dirty: bool,
    gpu_dirty: bool,
    sync_count: u32,
}

impl CoherenceTracker {
    fn new() -> Self {
        Self { cpu_dirty: false, gpu_dirty: false, sync_count: 0 }
    }

    fn cpu_write(&mut self) {
        self.cpu_dirty = true;
    }

    fn gpu_write(&mut self) {
        self.gpu_dirty = true;
    }

    fn sync_to_gpu(&mut self) -> bool {
        if self.cpu_dirty {
            self.cpu_dirty = false;
            self.sync_count += 1;
            true
        } else {
            false
        }
    }

    fn sync_to_cpu(&mut self) -> bool {
        if self.gpu_dirty {
            self.gpu_dirty = false;
            self.sync_count += 1;
            true
        } else {
            false
        }
    }
}

#[test]
fn storage_mode_private_for_gpu_only() {
    let mode = StorageModeSelector::select(false, false, true);
    assert_eq!(mode, StorageMode::Private);
}

#[test]
fn storage_mode_shared_for_readback() {
    let mode = StorageModeSelector::select(true, true, false);
    assert_eq!(mode, StorageMode::Shared);
}

#[test]
fn storage_mode_managed_for_gpu_write_cpu_read() {
    let mode = StorageModeSelector::select(true, false, true);
    assert_eq!(mode, StorageMode::Managed);
}

#[test]
fn coherence_no_sync_when_clean() {
    let mut tracker = CoherenceTracker::new();
    assert!(!tracker.sync_to_gpu());
    assert!(!tracker.sync_to_cpu());
    assert_eq!(tracker.sync_count, 0);
}

#[test]
fn coherence_cpu_write_needs_gpu_sync() {
    let mut tracker = CoherenceTracker::new();
    tracker.cpu_write();
    assert!(tracker.sync_to_gpu());
    assert_eq!(tracker.sync_count, 1);
    assert!(!tracker.sync_to_gpu()); // already synced
}

#[test]
fn coherence_gpu_write_needs_cpu_sync() {
    let mut tracker = CoherenceTracker::new();
    tracker.gpu_write();
    assert!(tracker.sync_to_cpu());
    assert_eq!(tracker.sync_count, 1);
}

#[test]
fn coherence_bidirectional_sync() {
    let mut tracker = CoherenceTracker::new();
    tracker.cpu_write();
    tracker.gpu_write();
    assert!(tracker.sync_to_gpu());
    assert!(tracker.sync_to_cpu());
    assert_eq!(tracker.sync_count, 2);
}

#[test]
fn shared_storage_zero_copy_semantics() {
    // On Apple Silicon UMA, Shared buffers have identical CPU/GPU pointers.
    let buf = MockBuffer::new(1024, StorageMode::Shared, "uma_buf");
    assert_eq!(buf.storage_mode, StorageMode::Shared);
    // No separate staging copy needed—validate size matches.
    assert!(buf.allocated_size >= 1024);
    assert_eq!(buf.contents.len(), buf.allocated_size);
}

// ══════════════════════════════════════════════════════════════════════
// 4. Buffer Lifecycle
// ══════════════════════════════════════════════════════════════════════

#[derive(Debug, PartialEq, Eq)]
enum LifecycleStage {
    Created,
    Written,
    Dispatched,
    Read,
    Released,
}

struct LifecycleBuffer {
    inner: MockBuffer,
    stage: LifecycleStage,
}

impl LifecycleBuffer {
    fn create(size: usize) -> Self {
        Self {
            inner: MockBuffer::new(size, StorageMode::Shared, "lifecycle"),
            stage: LifecycleStage::Created,
        }
    }

    fn write(&mut self, data: &[u8]) -> bool {
        if self.stage != LifecycleStage::Created && self.stage != LifecycleStage::Read {
            return false;
        }
        let len = data.len().min(self.inner.allocated_size);
        self.inner.contents[..len].copy_from_slice(&data[..len]);
        self.stage = LifecycleStage::Written;
        true
    }

    fn dispatch(&mut self) -> bool {
        if self.stage != LifecycleStage::Written {
            return false;
        }
        self.stage = LifecycleStage::Dispatched;
        true
    }

    fn read(&mut self) -> Option<&[u8]> {
        if self.stage != LifecycleStage::Dispatched {
            return None;
        }
        self.stage = LifecycleStage::Read;
        Some(&self.inner.contents[..self.inner.requested_size])
    }

    fn release(mut self) -> LifecycleStage {
        self.stage = LifecycleStage::Released;
        self.stage
    }
}

struct RefCountedBuffer {
    buffer: MockBuffer,
    ref_count: u32,
}

impl RefCountedBuffer {
    fn new(size: usize) -> Self {
        Self { buffer: MockBuffer::new(size, StorageMode::Shared, "refcounted"), ref_count: 1 }
    }

    fn retain(&mut self) {
        self.ref_count += 1;
    }

    fn release(&mut self) -> bool {
        self.ref_count -= 1;
        self.ref_count == 0
    }
}

/// Deferred deallocation queue.
struct DeferredDeallocator {
    pending: VecDeque<MockBuffer>,
    frame_delay: u32,
    current_frame: u32,
    frame_stamps: VecDeque<u32>,
}

impl DeferredDeallocator {
    fn new(frame_delay: u32) -> Self {
        Self {
            pending: VecDeque::new(),
            frame_delay,
            current_frame: 0,
            frame_stamps: VecDeque::new(),
        }
    }

    fn defer_release(&mut self, buf: MockBuffer) {
        self.pending.push_back(buf);
        self.frame_stamps.push_back(self.current_frame);
    }

    fn advance_frame(&mut self) -> usize {
        self.current_frame += 1;
        let mut freed = 0;
        while let Some(&stamp) = self.frame_stamps.front() {
            if self.current_frame - stamp >= self.frame_delay {
                self.pending.pop_front();
                self.frame_stamps.pop_front();
                freed += 1;
            } else {
                break;
            }
        }
        freed
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[test]
fn lifecycle_create_write_dispatch_read_release() {
    let mut buf = LifecycleBuffer::create(64);
    assert_eq!(buf.stage, LifecycleStage::Created);
    assert!(buf.write(&[1u8; 64]));
    assert_eq!(buf.stage, LifecycleStage::Written);
    assert!(buf.dispatch());
    assert_eq!(buf.stage, LifecycleStage::Dispatched);
    let data = buf.read().unwrap();
    assert_eq!(data.len(), 64);
    assert_eq!(buf.stage, LifecycleStage::Read);
    let final_stage = buf.release();
    assert_eq!(final_stage, LifecycleStage::Released);
}

#[test]
fn lifecycle_cannot_dispatch_without_write() {
    let mut buf = LifecycleBuffer::create(32);
    assert!(!buf.dispatch());
}

#[test]
fn lifecycle_cannot_read_without_dispatch() {
    let mut buf = LifecycleBuffer::create(32);
    assert!(buf.write(&[0u8; 32]));
    assert!(buf.read().is_none());
}

#[test]
fn lifecycle_ref_counting() {
    let mut buf = RefCountedBuffer::new(256);
    buf.retain();
    buf.retain();
    assert_eq!(buf.ref_count, 3);
    assert!(!buf.release());
    assert!(!buf.release());
    assert!(buf.release()); // last ref -> can free
}

#[test]
fn lifecycle_deferred_dealloc_respects_frame_delay() {
    let mut dealloc = DeferredDeallocator::new(3);
    dealloc.defer_release(MockBuffer::new(64, StorageMode::Shared, "deferred"));
    assert_eq!(dealloc.pending_count(), 1);
    assert_eq!(dealloc.advance_frame(), 0); // frame 1
    assert_eq!(dealloc.advance_frame(), 0); // frame 2
    assert_eq!(dealloc.advance_frame(), 1); // frame 3 → freed
    assert_eq!(dealloc.pending_count(), 0);
}

#[test]
fn lifecycle_deferred_multiple_buffers() {
    let mut dealloc = DeferredDeallocator::new(2);
    dealloc.defer_release(MockBuffer::new(32, StorageMode::Shared, "a"));
    dealloc.advance_frame(); // frame 1
    dealloc.defer_release(MockBuffer::new(32, StorageMode::Shared, "b"));
    assert_eq!(dealloc.pending_count(), 2);
    assert_eq!(dealloc.advance_frame(), 1); // frame 2: frees "a"
    assert_eq!(dealloc.pending_count(), 1);
    assert_eq!(dealloc.advance_frame(), 1); // frame 3: frees "b"
    assert_eq!(dealloc.pending_count(), 0);
}

#[test]
fn lifecycle_rewrite_after_read() {
    let mut buf = LifecycleBuffer::create(16);
    assert!(buf.write(&[1u8; 16]));
    assert!(buf.dispatch());
    let _ = buf.read().unwrap();
    // After read, buffer can be rewritten
    assert!(buf.write(&[2u8; 16]));
    assert_eq!(buf.stage, LifecycleStage::Written);
}

#[test]
fn lifecycle_data_integrity() {
    let mut buf = LifecycleBuffer::create(8);
    let data = [10, 20, 30, 40, 50, 60, 70, 80];
    assert!(buf.write(&data));
    assert!(buf.dispatch());
    let readback = buf.read().unwrap();
    assert_eq!(readback, &data);
}

// ══════════════════════════════════════════════════════════════════════
// 5. Memory Pressure Handling
// ══════════════════════════════════════════════════════════════════════

struct PressureManager {
    buffers: Vec<(BufferId, usize, BufferPriority, u64)>, // (id, size, priority, last_access)
    total_budget: usize,
    used: usize,
    eviction_policy: EvictionPolicy,
    tick: u64,
    eviction_count: u64,
    pressure_callbacks_fired: u32,
}

impl PressureManager {
    fn new(budget: usize, policy: EvictionPolicy) -> Self {
        Self {
            buffers: Vec::new(),
            total_budget: budget,
            used: 0,
            eviction_policy: policy,
            tick: 0,
            eviction_count: 0,
            pressure_callbacks_fired: 0,
        }
    }

    fn allocate(&mut self, size: usize, priority: BufferPriority) -> Option<BufferId> {
        while self.used + size > self.total_budget {
            if !self.evict_one() {
                self.pressure_callbacks_fired += 1;
                return None;
            }
        }
        self.tick += 1;
        let id = next_buffer_id();
        self.buffers.push((id, size, priority, self.tick));
        self.used += size;
        Some(id)
    }

    fn touch(&mut self, id: BufferId) {
        self.tick += 1;
        if let Some(entry) = self.buffers.iter_mut().find(|e| e.0 == id) {
            entry.3 = self.tick;
        }
    }

    fn evict_one(&mut self) -> bool {
        if self.buffers.is_empty() {
            return false;
        }
        let idx = match self.eviction_policy {
            EvictionPolicy::Lru => {
                // Evict least-recently-used (lowest tick)
                self.buffers.iter().enumerate().min_by_key(|(_, e)| e.3).map(|(i, _)| i)
            }
            EvictionPolicy::Priority => {
                // Evict lowest priority, then LRU within same priority
                self.buffers.iter().enumerate().min_by_key(|(_, e)| (e.2, e.3)).map(|(i, _)| i)
            }
        };
        if let Some(i) = idx {
            let (_, size, _, _) = self.buffers.remove(i);
            self.used -= size;
            self.eviction_count += 1;
            true
        } else {
            false
        }
    }

    fn pressure_level(&self) -> f64 {
        if self.total_budget == 0 {
            return 1.0;
        }
        self.used as f64 / self.total_budget as f64
    }
}

#[test]
fn pressure_lru_eviction() {
    let mut mgr = PressureManager::new(1024, EvictionPolicy::Lru);
    let id1 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    let _id2 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    // Pool full; next allocation must evict id1 (oldest)
    mgr.touch(_id2);
    let _id3 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    assert_eq!(mgr.eviction_count, 1);
    assert!(!mgr.buffers.iter().any(|e| e.0 == id1));
}

#[test]
fn pressure_priority_eviction() {
    let mut mgr = PressureManager::new(1024, EvictionPolicy::Priority);
    let id_low = mgr.allocate(512, BufferPriority::Low).unwrap();
    let _id_high = mgr.allocate(512, BufferPriority::High).unwrap();
    let _id3 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    assert_eq!(mgr.eviction_count, 1);
    // Low-priority buffer should be evicted
    assert!(!mgr.buffers.iter().any(|e| e.0 == id_low));
}

#[test]
fn pressure_callback_when_unrecoverable() {
    let mut mgr = PressureManager::new(256, EvictionPolicy::Lru);
    let _id = mgr.allocate(256, BufferPriority::Critical).unwrap();
    // Can't evict Critical, so next allocation fails and fires callback
    // Actually, our eviction doesn't check priority for Lru — let's just fill
    // and try to allocate more than budget.
    let result = mgr.allocate(512, BufferPriority::Normal);
    // After evicting 256 we still only have 256 < 512
    assert!(result.is_none());
    assert!(mgr.pressure_callbacks_fired > 0);
}

#[test]
fn pressure_graceful_degradation() {
    let mut mgr = PressureManager::new(1024, EvictionPolicy::Lru);
    for i in 0..4 {
        mgr.allocate(256, BufferPriority::Normal).unwrap();
        let _ = i;
    }
    assert!(mgr.pressure_level() > 0.99);
    // Allocating beyond triggers eviction, not panic
    let id = mgr.allocate(256, BufferPriority::Normal);
    assert!(id.is_some());
    assert!(mgr.eviction_count >= 1);
}

#[test]
fn pressure_level_calculation() {
    let mut mgr = PressureManager::new(1000, EvictionPolicy::Lru);
    assert!((mgr.pressure_level() - 0.0).abs() < 0.001);
    mgr.allocate(500, BufferPriority::Normal).unwrap();
    assert!((mgr.pressure_level() - 0.5).abs() < 0.001);
}

#[test]
fn pressure_touch_prevents_eviction() {
    let mut mgr = PressureManager::new(1024, EvictionPolicy::Lru);
    let id1 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    let _id2 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    // Touch id1 to make it most recent
    mgr.touch(id1);
    let _id3 = mgr.allocate(512, BufferPriority::Normal).unwrap();
    // id2 should have been evicted (now oldest), id1 survives
    assert!(mgr.buffers.iter().any(|e| e.0 == id1));
}

#[test]
fn pressure_multiple_evictions() {
    let mut mgr = PressureManager::new(512, EvictionPolicy::Lru);
    for _ in 0..4 {
        mgr.allocate(128, BufferPriority::Normal).unwrap();
    }
    // Need 384 → must evict 3 buffers (3×128)
    let result = mgr.allocate(384, BufferPriority::Normal);
    assert!(result.is_some());
    assert_eq!(mgr.eviction_count, 3);
}

#[test]
fn pressure_empty_pool_allocation() {
    let mut mgr = PressureManager::new(0, EvictionPolicy::Lru);
    let result = mgr.allocate(1, BufferPriority::Normal);
    assert!(result.is_none());
    assert!(mgr.pressure_callbacks_fired > 0);
}

// ══════════════════════════════════════════════════════════════════════
// 6. Transfer Optimization
// ══════════════════════════════════════════════════════════════════════

/// Simulated blit (copy) encoder operation.
#[derive(Debug, Clone)]
struct BlitOp {
    src: BufferId,
    dst: BufferId,
    size: usize,
    offset_src: usize,
    offset_dst: usize,
}

struct BlitEncoder {
    ops: Vec<BlitOp>,
    committed: bool,
}

impl BlitEncoder {
    fn new() -> Self {
        Self { ops: Vec::new(), committed: false }
    }

    fn copy(&mut self, src: BufferId, dst: BufferId, size: usize) {
        self.ops.push(BlitOp { src, dst, size, offset_src: 0, offset_dst: 0 });
    }

    fn copy_with_offsets(
        &mut self,
        src: BufferId,
        src_off: usize,
        dst: BufferId,
        dst_off: usize,
        size: usize,
    ) {
        self.ops.push(BlitOp { src, dst, size, offset_src: src_off, offset_dst: dst_off });
    }

    fn commit(&mut self) {
        self.committed = true;
    }

    fn total_bytes(&self) -> usize {
        self.ops.iter().map(|o| o.size).sum()
    }
}

/// Staging buffer for host→device transfers.
struct StagingBuffer {
    buffer: MockBuffer,
    write_offset: usize,
}

impl StagingBuffer {
    fn new(capacity: usize) -> Self {
        Self { buffer: MockBuffer::new(capacity, StorageMode::Shared, "staging"), write_offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.buffer.allocated_size - self.write_offset
    }

    fn stage(&mut self, data: &[u8]) -> Option<usize> {
        if data.len() > self.remaining() {
            return None;
        }
        let offset = self.write_offset;
        self.buffer.contents[offset..offset + data.len()].copy_from_slice(data);
        self.write_offset += data.len();
        Some(offset)
    }

    fn reset(&mut self) {
        self.write_offset = 0;
    }
}

/// Simulated async copy tracker.
struct AsyncCopyTracker {
    pending: VecDeque<(BlitOp, bool)>, // (op, completed)
}

impl AsyncCopyTracker {
    fn new() -> Self {
        Self { pending: VecDeque::new() }
    }

    fn submit(&mut self, op: BlitOp) {
        self.pending.push_back((op, false));
    }

    fn complete_next(&mut self) -> bool {
        for entry in self.pending.iter_mut() {
            if !entry.1 {
                entry.1 = true;
                return true;
            }
        }
        false
    }

    fn drain_completed(&mut self) -> usize {
        let before = self.pending.len();
        self.pending.retain(|(_, done)| !done);
        before - self.pending.len()
    }

    fn pending_count(&self) -> usize {
        self.pending.iter().filter(|(_, done)| !done).count()
    }
}

#[test]
fn transfer_blit_encoder_basic() {
    let mut enc = BlitEncoder::new();
    let src = next_buffer_id();
    let dst = next_buffer_id();
    enc.copy(src, dst, 4096);
    enc.commit();
    assert!(enc.committed);
    assert_eq!(enc.ops.len(), 1);
    assert_eq!(enc.total_bytes(), 4096);
}

#[test]
fn transfer_blit_with_offsets() {
    let mut enc = BlitEncoder::new();
    let src = next_buffer_id();
    let dst = next_buffer_id();
    enc.copy_with_offsets(src, 256, dst, 512, 1024);
    assert_eq!(enc.ops[0].offset_src, 256);
    assert_eq!(enc.ops[0].offset_dst, 512);
}

#[test]
fn transfer_async_copy_completion() {
    let mut tracker = AsyncCopyTracker::new();
    let op = BlitOp {
        src: next_buffer_id(),
        dst: next_buffer_id(),
        size: 2048,
        offset_src: 0,
        offset_dst: 0,
    };
    tracker.submit(op);
    assert_eq!(tracker.pending_count(), 1);
    assert!(tracker.complete_next());
    assert_eq!(tracker.drain_completed(), 1);
    assert_eq!(tracker.pending_count(), 0);
}

#[test]
fn transfer_staging_buffer_fill() {
    let mut staging = StagingBuffer::new(1024);
    let off1 = staging.stage(&[1u8; 512]).unwrap();
    assert_eq!(off1, 0);
    let off2 = staging.stage(&[2u8; 512]).unwrap();
    assert_eq!(off2, 512);
    assert!(staging.stage(&[3u8; 1]).is_none()); // full
}

#[test]
fn transfer_staging_reset() {
    let mut staging = StagingBuffer::new(256);
    staging.stage(&[0u8; 256]).unwrap();
    assert_eq!(staging.remaining(), 0);
    staging.reset();
    assert_eq!(staging.remaining(), 256);
}

#[test]
fn transfer_multiple_blits_batch() {
    let mut enc = BlitEncoder::new();
    for _ in 0..10 {
        enc.copy(next_buffer_id(), next_buffer_id(), 128);
    }
    assert_eq!(enc.ops.len(), 10);
    assert_eq!(enc.total_bytes(), 1280);
}

#[test]
fn transfer_async_multiple_pending() {
    let mut tracker = AsyncCopyTracker::new();
    for _ in 0..5 {
        tracker.submit(BlitOp {
            src: next_buffer_id(),
            dst: next_buffer_id(),
            size: 64,
            offset_src: 0,
            offset_dst: 0,
        });
    }
    assert_eq!(tracker.pending_count(), 5);
    tracker.complete_next();
    tracker.complete_next();
    assert_eq!(tracker.drain_completed(), 2);
    assert_eq!(tracker.pending_count(), 3);
}

#[test]
fn transfer_staging_data_integrity() {
    let mut staging = StagingBuffer::new(64);
    let data = [0xDE, 0xAD, 0xBE, 0xEF];
    let off = staging.stage(&data).unwrap();
    assert_eq!(&staging.buffer.contents[off..off + 4], &data);
}

// ══════════════════════════════════════════════════════════════════════
// 7. Texture Memory
// ══════════════════════════════════════════════════════════════════════

#[test]
fn texture_2d_allocation_size() {
    let desc = TextureDescriptor {
        width: 256,
        height: 256,
        depth: 1,
        dimension: TextureDimension::D2,
        format: PixelFormat::Rgba8Unorm,
        mipmap_levels: 1,
    };
    assert_eq!(desc.base_size_bytes(), 256 * 256 * 4);
}

#[test]
fn texture_3d_allocation_size() {
    let desc = TextureDescriptor {
        width: 64,
        height: 64,
        depth: 32,
        dimension: TextureDimension::D3,
        format: PixelFormat::R32Float,
        mipmap_levels: 1,
    };
    assert_eq!(desc.base_size_bytes(), 64 * 64 * 32 * 4);
}

#[test]
fn texture_mipmap_storage_chain() {
    let desc = TextureDescriptor {
        width: 256,
        height: 256,
        depth: 1,
        dimension: TextureDimension::D2,
        format: PixelFormat::Rgba8Unorm,
        mipmap_levels: 4,
    };
    // level 0: 256×256×4 = 262144
    // level 1: 128×128×4 = 65536
    // level 2: 64×64×4   = 16384
    // level 3: 32×32×4   = 4096
    let total = desc.total_size_with_mipmaps();
    assert_eq!(total, 262144 + 65536 + 16384 + 4096);
}

#[test]
fn texture_full_mip_chain() {
    let desc = TextureDescriptor {
        width: 8,
        height: 8,
        depth: 1,
        dimension: TextureDimension::D2,
        format: PixelFormat::Rgba8Unorm,
        mipmap_levels: 4,
    };
    // 8×8×4=256, 4×4×4=64, 2×2×4=16, 1×1×4=4
    assert_eq!(desc.total_size_with_mipmaps(), 256 + 64 + 16 + 4);
}

#[test]
fn texture_format_conversion_cost() {
    // Compare storage cost across formats
    let base_size = 1024 * 1024; // 1M pixels
    let r32 = base_size * PixelFormat::R32Float.bytes_per_pixel();
    let rgba8 = base_size * PixelFormat::Rgba8Unorm.bytes_per_pixel();
    let rgba16f = base_size * PixelFormat::Rgba16Float.bytes_per_pixel();
    let rgba32f = base_size * PixelFormat::Rgba32Float.bytes_per_pixel();
    assert_eq!(r32, rgba8); // both 4 bytes/pixel
    assert_eq!(rgba16f, 2 * rgba8);
    assert_eq!(rgba32f, 4 * rgba8);
}

#[test]
fn texture_single_pixel_mipmap() {
    let desc = TextureDescriptor {
        width: 1,
        height: 1,
        depth: 1,
        dimension: TextureDimension::D2,
        format: PixelFormat::R32Float,
        mipmap_levels: 3,
    };
    // All mip levels are 1×1 (can't go below 1)
    assert_eq!(desc.total_size_with_mipmaps(), 4 + 4 + 4);
}

#[test]
fn texture_3d_with_mipmaps() {
    let desc = TextureDescriptor {
        width: 16,
        height: 16,
        depth: 4,
        dimension: TextureDimension::D3,
        format: PixelFormat::R32Float,
        mipmap_levels: 2,
    };
    // level 0: 16×16×4×4 = 4096
    // level 1: 8×8×4×4   = 1024  (depth stays for our simple mip)
    assert_eq!(desc.total_size_with_mipmaps(), 4096 + 1024);
}

#[test]
fn texture_dimension_mismatch_guard() {
    let desc = TextureDescriptor {
        width: 32,
        height: 32,
        depth: 1,
        dimension: TextureDimension::D2,
        format: PixelFormat::Rgba32Float,
        mipmap_levels: 1,
    };
    // 2D texture with depth=1 is fine
    assert_eq!(desc.base_size_bytes(), 32 * 32 * 16);
}

// ══════════════════════════════════════════════════════════════════════
// 8. Resource Heap Management
// ══════════════════════════════════════════════════════════════════════

struct HeapAllocation {
    offset: usize,
    size: usize,
    id: BufferId,
}

struct ResourceHeap {
    capacity: usize,
    allocations: Vec<HeapAllocation>,
    free_offset: usize,
    default_alignment: usize,
}

impl ResourceHeap {
    fn new(capacity: usize, alignment: usize) -> Self {
        Self { capacity, allocations: Vec::new(), free_offset: 0, default_alignment: alignment }
    }

    fn sub_allocate(&mut self, size: usize) -> Option<BufferId> {
        self.sub_allocate_aligned(size, self.default_alignment)
    }

    fn sub_allocate_aligned(&mut self, size: usize, alignment: usize) -> Option<BufferId> {
        let aligned_offset = align_up(self.free_offset, alignment);
        let aligned_size = align_up(size, alignment);
        if aligned_offset + aligned_size > self.capacity {
            return None;
        }
        let id = next_buffer_id();
        self.allocations.push(HeapAllocation { offset: aligned_offset, size: aligned_size, id });
        self.free_offset = aligned_offset + aligned_size;
        Some(id)
    }

    fn used(&self) -> usize {
        self.free_offset
    }

    fn remaining(&self) -> usize {
        self.capacity - self.free_offset
    }

    fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    fn fragmentation_ratio(&self) -> f64 {
        if self.allocations.is_empty() || self.capacity == 0 {
            return 0.0;
        }
        let actual_data: usize = self.allocations.iter().map(|a| a.size).sum();
        let wasted = self.free_offset - actual_data;
        wasted as f64 / self.free_offset as f64
    }
}

#[test]
fn heap_creation() {
    let heap = ResourceHeap::new(1024 * 1024, 256);
    assert_eq!(heap.capacity, 1024 * 1024);
    assert_eq!(heap.used(), 0);
    assert_eq!(heap.remaining(), 1024 * 1024);
}

#[test]
fn heap_sub_allocation() {
    let mut heap = ResourceHeap::new(4096, 256);
    let id = heap.sub_allocate(512).unwrap();
    assert_eq!(heap.allocation_count(), 1);
    assert_eq!(heap.used(), 512);
    let _ = id;
}

#[test]
fn heap_multiple_sub_allocations() {
    let mut heap = ResourceHeap::new(4096, 256);
    heap.sub_allocate(256).unwrap();
    heap.sub_allocate(512).unwrap();
    heap.sub_allocate(256).unwrap();
    assert_eq!(heap.allocation_count(), 3);
    assert_eq!(heap.used(), 1024);
}

#[test]
fn heap_exhaustion() {
    let mut heap = ResourceHeap::new(512, 256);
    heap.sub_allocate(256).unwrap();
    heap.sub_allocate(256).unwrap();
    assert!(heap.sub_allocate(256).is_none());
}

#[test]
fn heap_type_aligned_allocation() {
    let mut heap = ResourceHeap::new(8192, 16);
    // Buffer alignment (256 bytes)
    let _id1 = heap.sub_allocate_aligned(100, 256).unwrap();
    assert_eq!(heap.allocations[0].offset % 256, 0);

    // Texture alignment (4096 bytes — page aligned)
    let _id2 = heap.sub_allocate_aligned(1024, 4096).unwrap();
    assert_eq!(heap.allocations[1].offset % 4096, 0);
}

#[test]
fn heap_alignment_padding_tracked() {
    let mut heap = ResourceHeap::new(16384, 16);
    // Allocate a small amount, then request page-aligned
    heap.sub_allocate_aligned(32, 16).unwrap();
    heap.sub_allocate_aligned(64, 4096).unwrap();
    // Second allocation starts at 4096 (padded from offset 32), so fragmentation > 0
    assert!(heap.fragmentation_ratio() > 0.0);
}

#[test]
fn heap_fragmentation_zero_when_dense() {
    let mut heap = ResourceHeap::new(1024, 256);
    heap.sub_allocate(256).unwrap();
    heap.sub_allocate(256).unwrap();
    // All allocations are tightly packed at 256-byte alignment
    assert!((heap.fragmentation_ratio() - 0.0).abs() < 0.001);
}

#[test]
fn heap_remaining_accuracy() {
    let mut heap = ResourceHeap::new(2048, 256);
    heap.sub_allocate(768).unwrap(); // rounds to 768 (already aligned)
    assert_eq!(heap.remaining(), 2048 - 768);
}

// ══════════════════════════════════════════════════════════════════════
// 9. Ring Buffer
// ══════════════════════════════════════════════════════════════════════

/// Ring buffer for streaming uploads to GPU.
struct RingBuffer {
    capacity: usize,
    write_head: usize,
    read_head: usize,
    frames_in_flight: VecDeque<(usize, usize)>, // (start, end) per frame
    max_frames: usize,
}

impl RingBuffer {
    fn new(capacity: usize, max_frames: usize) -> Self {
        Self {
            capacity,
            write_head: 0,
            read_head: 0,
            frames_in_flight: VecDeque::new(),
            max_frames,
        }
    }

    fn available(&self) -> usize {
        if self.write_head >= self.read_head {
            self.capacity - (self.write_head - self.read_head)
        } else {
            self.read_head - self.write_head
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        let aligned = align_up(size, 256);
        if aligned > self.available() {
            return None;
        }
        let offset = self.write_head;
        let new_head = self.write_head + aligned;
        if new_head <= self.capacity {
            self.write_head = new_head;
            Some(offset)
        } else if aligned <= self.read_head {
            // Wrap around
            self.write_head = aligned;
            Some(0)
        } else {
            None
        }
    }

    fn begin_frame(&mut self) {
        if self.frames_in_flight.len() >= self.max_frames {
            self.retire_oldest_frame();
        }
    }

    fn end_frame(&mut self) {
        self.frames_in_flight.push_back((self.read_head, self.write_head));
    }

    fn retire_oldest_frame(&mut self) -> bool {
        if let Some((_, end)) = self.frames_in_flight.pop_front() {
            self.read_head = end;
            true
        } else {
            false
        }
    }

    fn frames_in_flight_count(&self) -> usize {
        self.frames_in_flight.len()
    }
}

#[test]
fn ring_buffer_basic_allocation() {
    let mut ring = RingBuffer::new(4096, 3);
    let off = ring.allocate(256).unwrap();
    assert_eq!(off, 0);
    assert_eq!(ring.available(), 4096 - 256);
}

#[test]
fn ring_buffer_sequential_allocs() {
    let mut ring = RingBuffer::new(1024, 3);
    let o1 = ring.allocate(256).unwrap();
    let o2 = ring.allocate(256).unwrap();
    assert_eq!(o1, 0);
    assert_eq!(o2, 256);
}

#[test]
fn ring_buffer_wrap_around() {
    let mut ring = RingBuffer::new(1024, 3);
    // Fill most of the buffer
    ring.allocate(768).unwrap();
    ring.end_frame();
    ring.retire_oldest_frame();
    // read_head is now at 768, write_head at 768
    // Allocate something that won't fit at the end → wraps to beginning
    let off = ring.allocate(512).unwrap();
    assert_eq!(off, 0); // wrapped
}

#[test]
fn ring_buffer_frame_tracking() {
    let mut ring = RingBuffer::new(4096, 3);
    ring.allocate(512).unwrap();
    ring.end_frame();
    ring.allocate(512).unwrap();
    ring.end_frame();
    assert_eq!(ring.frames_in_flight_count(), 2);
}

#[test]
fn ring_buffer_retire_frees_space() {
    let mut ring = RingBuffer::new(1024, 3);
    ring.allocate(512).unwrap();
    ring.end_frame();
    ring.allocate(512).unwrap();
    ring.end_frame();
    assert!(ring.allocate(256).is_none()); // full
    ring.retire_oldest_frame();
    assert!(ring.allocate(256).is_some()); // space freed
}

#[test]
fn ring_buffer_max_frames_auto_retire() {
    let mut ring = RingBuffer::new(4096, 2);
    ring.allocate(512).unwrap();
    ring.end_frame();
    ring.allocate(512).unwrap();
    ring.end_frame();
    // begin_frame retires oldest if at max
    ring.begin_frame();
    assert_eq!(ring.frames_in_flight_count(), 1);
}

#[test]
fn ring_buffer_exhaustion() {
    let mut ring = RingBuffer::new(512, 3);
    ring.allocate(512).unwrap();
    ring.end_frame();
    assert!(ring.allocate(256).is_none());
}

#[test]
fn ring_buffer_alignment() {
    let mut ring = RingBuffer::new(4096, 3);
    let o1 = ring.allocate(100).unwrap(); // rounds to 256
    let o2 = ring.allocate(100).unwrap();
    assert_eq!(o1, 0);
    assert_eq!(o2, 256); // aligned to 256
}

// ══════════════════════════════════════════════════════════════════════
// 10. Memory Statistics
// ══════════════════════════════════════════════════════════════════════

struct MemoryStats {
    current_allocated: usize,
    peak_allocated: usize,
    total_allocations: u64,
    total_deallocations: u64,
    bytes_allocated_lifetime: u64,
    bytes_freed_lifetime: u64,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            current_allocated: 0,
            peak_allocated: 0,
            total_allocations: 0,
            total_deallocations: 0,
            bytes_allocated_lifetime: 0,
            bytes_freed_lifetime: 0,
        }
    }

    fn record_alloc(&mut self, size: usize) {
        self.current_allocated += size;
        self.bytes_allocated_lifetime += size as u64;
        self.total_allocations += 1;
        if self.current_allocated > self.peak_allocated {
            self.peak_allocated = self.current_allocated;
        }
    }

    fn record_free(&mut self, size: usize) {
        self.current_allocated -= size;
        self.bytes_freed_lifetime += size as u64;
        self.total_deallocations += 1;
    }

    fn fragmentation_ratio(&self) -> f64 {
        if self.peak_allocated == 0 {
            return 0.0;
        }
        1.0 - (self.current_allocated as f64 / self.peak_allocated as f64)
    }

    fn bandwidth_estimate_gbps(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            return 0.0;
        }
        let total_bytes = self.bytes_allocated_lifetime + self.bytes_freed_lifetime;
        (total_bytes as f64) / elapsed_secs / 1e9
    }

    fn leak_check(&self) -> bool {
        self.total_allocations == self.total_deallocations && self.current_allocated == 0
    }
}

#[test]
fn stats_peak_tracking() {
    let mut stats = MemoryStats::new();
    stats.record_alloc(1024);
    stats.record_alloc(2048);
    assert_eq!(stats.peak_allocated, 3072);
    stats.record_free(2048);
    assert_eq!(stats.peak_allocated, 3072); // peak unchanged
    assert_eq!(stats.current_allocated, 1024);
}

#[test]
fn stats_current_allocation() {
    let mut stats = MemoryStats::new();
    stats.record_alloc(512);
    stats.record_alloc(256);
    stats.record_free(512);
    assert_eq!(stats.current_allocated, 256);
}

#[test]
fn stats_fragmentation_ratio() {
    let mut stats = MemoryStats::new();
    stats.record_alloc(1000);
    stats.record_free(750);
    // peak=1000, current=250 → frag = 1 - 250/1000 = 0.75
    assert!((stats.fragmentation_ratio() - 0.75).abs() < 0.001);
}

#[test]
fn stats_bandwidth_estimate() {
    let mut stats = MemoryStats::new();
    stats.record_alloc(1_000_000_000); // 1 GB
    let bw = stats.bandwidth_estimate_gbps(1.0);
    assert!((bw - 1.0).abs() < 0.001);
}

#[test]
fn stats_allocation_count() {
    let mut stats = MemoryStats::new();
    for _ in 0..100 {
        stats.record_alloc(64);
    }
    for _ in 0..50 {
        stats.record_free(64);
    }
    assert_eq!(stats.total_allocations, 100);
    assert_eq!(stats.total_deallocations, 50);
}

#[test]
fn stats_leak_check_clean() {
    let mut stats = MemoryStats::new();
    stats.record_alloc(256);
    stats.record_free(256);
    assert!(stats.leak_check());
}

#[test]
fn stats_leak_check_detects_leak() {
    let mut stats = MemoryStats::new();
    stats.record_alloc(256);
    assert!(!stats.leak_check());
}

#[test]
fn stats_zero_initial() {
    let stats = MemoryStats::new();
    assert_eq!(stats.current_allocated, 0);
    assert_eq!(stats.peak_allocated, 0);
    assert_eq!(stats.total_allocations, 0);
    assert!((stats.fragmentation_ratio() - 0.0).abs() < 0.001);
}

// ══════════════════════════════════════════════════════════════════════
// 11. Apple Silicon Unified Memory
// ══════════════════════════════════════════════════════════════════════

/// UMA buffer that models zero-copy CPU↔GPU access on Apple Silicon.
struct UmaBuffer {
    data: Vec<u8>,
    cpu_ptr_valid: bool,
    gpu_ptr_valid: bool,
    storage_mode: StorageMode,
}

impl UmaBuffer {
    fn new_shared(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            cpu_ptr_valid: true,
            gpu_ptr_valid: true,
            storage_mode: StorageMode::Shared,
        }
    }

    fn new_private(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            cpu_ptr_valid: false,
            gpu_ptr_valid: true,
            storage_mode: StorageMode::Private,
        }
    }

    fn is_zero_copy(&self) -> bool {
        self.storage_mode == StorageMode::Shared && self.cpu_ptr_valid && self.gpu_ptr_valid
    }

    fn cpu_write(&mut self, offset: usize, data: &[u8]) -> bool {
        if !self.cpu_ptr_valid {
            return false;
        }
        if offset + data.len() > self.data.len() {
            return false;
        }
        self.data[offset..offset + data.len()].copy_from_slice(data);
        true
    }

    fn gpu_read(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if !self.gpu_ptr_valid {
            return None;
        }
        if offset + len > self.data.len() {
            return None;
        }
        Some(&self.data[offset..offset + len])
    }
}

/// Model bandwidth between CPU and GPU on UMA.
struct UmaBandwidthModel {
    /// Peak memory bandwidth in GB/s (M1: ~68, M2: ~100, M3: ~150, M4: ~273).
    peak_bandwidth_gbps: f64,
    /// Typical efficiency ratio (0.0–1.0).
    efficiency: f64,
}

impl UmaBandwidthModel {
    fn m1() -> Self {
        Self { peak_bandwidth_gbps: 68.0, efficiency: 0.85 }
    }

    fn m4_max() -> Self {
        Self { peak_bandwidth_gbps: 273.0, efficiency: 0.80 }
    }

    fn effective_bandwidth(&self) -> f64 {
        self.peak_bandwidth_gbps * self.efficiency
    }

    fn transfer_time_ms(&self, bytes: usize) -> f64 {
        let gb = bytes as f64 / 1e9;
        (gb / self.effective_bandwidth()) * 1000.0
    }
}

#[test]
fn uma_zero_copy_shared() {
    let buf = UmaBuffer::new_shared(4096);
    assert!(buf.is_zero_copy());
}

#[test]
fn uma_private_not_cpu_accessible() {
    let buf = UmaBuffer::new_private(1024);
    assert!(!buf.is_zero_copy());
    assert!(!buf.cpu_ptr_valid);
}

#[test]
fn uma_coherent_write_read() {
    let mut buf = UmaBuffer::new_shared(256);
    let data = [42u8; 64];
    assert!(buf.cpu_write(0, &data));
    let readback = buf.gpu_read(0, 64).unwrap();
    assert_eq!(readback, &data);
}

#[test]
fn uma_private_rejects_cpu_write() {
    let mut buf = UmaBuffer::new_private(256);
    assert!(!buf.cpu_write(0, &[1, 2, 3]));
}

#[test]
fn uma_bandwidth_m1() {
    let model = UmaBandwidthModel::m1();
    let bw = model.effective_bandwidth();
    assert!(bw > 50.0 && bw < 68.0);
}

#[test]
fn uma_bandwidth_m4() {
    let model = UmaBandwidthModel::m4_max();
    assert!(model.effective_bandwidth() > 200.0);
}

#[test]
fn uma_transfer_time_estimation() {
    let model = UmaBandwidthModel::m1();
    let time_ms = model.transfer_time_ms(1_000_000_000); // 1 GB
    // ~68 * 0.85 = ~57.8 GB/s → 1 GB takes ~17.3 ms
    assert!(time_ms > 15.0 && time_ms < 25.0);
}

#[test]
fn uma_zero_copy_avoids_staging() {
    // On UMA with Shared storage, no staging buffer needed.
    let buf = UmaBuffer::new_shared(1024);
    assert!(buf.is_zero_copy());
    // Verify we can write+read without any copy step
    let mut buf = buf;
    assert!(buf.cpu_write(0, &[0xFF; 8]));
    // Direct read verifies same memory
    assert_eq!(buf.gpu_read(0, 8).unwrap(), &[0xFF; 8]);
}

// ══════════════════════════════════════════════════════════════════════
// 12. Regression Detection
// ══════════════════════════════════════════════════════════════════════

/// Tracks allocation sizes to detect size drift across versions.
struct AllocationSizeTracker {
    baseline: HashMap<String, usize>,
    current: HashMap<String, usize>,
    tolerance_pct: f64,
}

impl AllocationSizeTracker {
    fn new(tolerance_pct: f64) -> Self {
        Self { baseline: HashMap::new(), current: HashMap::new(), tolerance_pct }
    }

    fn set_baseline(&mut self, name: &str, size: usize) {
        self.baseline.insert(name.to_string(), size);
    }

    fn record(&mut self, name: &str, size: usize) {
        self.current.insert(name.to_string(), size);
    }

    fn check_drift(&self) -> Vec<(String, usize, usize, f64)> {
        let mut drifts = Vec::new();
        for (name, &current) in &self.current {
            if let Some(&baseline) = self.baseline.get(name) {
                if baseline == 0 {
                    continue;
                }
                let pct = ((current as f64 - baseline as f64) / baseline as f64).abs() * 100.0;
                if pct > self.tolerance_pct {
                    drifts.push((name.clone(), baseline, current, pct));
                }
            }
        }
        drifts
    }
}

/// Double-free guard using a set of valid buffer IDs.
struct DoubleFreeGuard {
    live: std::collections::HashSet<BufferId>,
    double_free_detected: bool,
}

impl DoubleFreeGuard {
    fn new() -> Self {
        Self { live: std::collections::HashSet::new(), double_free_detected: false }
    }

    fn register(&mut self, id: BufferId) {
        self.live.insert(id);
    }

    fn release(&mut self, id: BufferId) -> bool {
        if self.live.remove(&id) {
            true
        } else {
            self.double_free_detected = true;
            false
        }
    }
}

/// Leak detector: tracks alloc/free pairs.
struct LeakDetector {
    outstanding: HashMap<BufferId, String>,
}

impl LeakDetector {
    fn new() -> Self {
        Self { outstanding: HashMap::new() }
    }

    fn track_alloc(&mut self, id: BufferId, label: &str) {
        self.outstanding.insert(id, label.to_string());
    }

    fn track_free(&mut self, id: BufferId) -> bool {
        self.outstanding.remove(&id).is_some()
    }

    fn leaked_buffers(&self) -> Vec<(&BufferId, &String)> {
        self.outstanding.iter().collect()
    }

    fn has_leaks(&self) -> bool {
        !self.outstanding.is_empty()
    }
}

#[test]
fn regression_size_drift_detection() {
    let mut tracker = AllocationSizeTracker::new(5.0); // 5% tolerance
    tracker.set_baseline("weights", 1000);
    tracker.record("weights", 1080); // 8% drift
    let drifts = tracker.check_drift();
    assert_eq!(drifts.len(), 1);
    assert!(drifts[0].3 > 5.0);
}

#[test]
fn regression_no_drift_within_tolerance() {
    let mut tracker = AllocationSizeTracker::new(10.0);
    tracker.set_baseline("kv_cache", 2048);
    tracker.record("kv_cache", 2100); // ~2.5%
    assert!(tracker.check_drift().is_empty());
}

#[test]
fn regression_leak_detection() {
    let mut detector = LeakDetector::new();
    let id1 = next_buffer_id();
    let id2 = next_buffer_id();
    detector.track_alloc(id1, "weights");
    detector.track_alloc(id2, "activations");
    detector.track_free(id1);
    assert!(detector.has_leaks());
    assert_eq!(detector.leaked_buffers().len(), 1);
}

#[test]
fn regression_no_leaks_when_all_freed() {
    let mut detector = LeakDetector::new();
    let id = next_buffer_id();
    detector.track_alloc(id, "temp");
    detector.track_free(id);
    assert!(!detector.has_leaks());
}

#[test]
fn regression_double_free_detected() {
    let mut guard = DoubleFreeGuard::new();
    let id = next_buffer_id();
    guard.register(id);
    assert!(guard.release(id));
    assert!(!guard.release(id)); // double free
    assert!(guard.double_free_detected);
}

#[test]
fn regression_double_free_false_when_normal() {
    let mut guard = DoubleFreeGuard::new();
    let id = next_buffer_id();
    guard.register(id);
    guard.release(id);
    assert!(!guard.double_free_detected);
}

#[test]
fn regression_multiple_baselines() {
    let mut tracker = AllocationSizeTracker::new(5.0);
    tracker.set_baseline("a", 100);
    tracker.set_baseline("b", 200);
    tracker.set_baseline("c", 300);
    tracker.record("a", 100);
    tracker.record("b", 200);
    tracker.record("c", 400); // 33% drift
    let drifts = tracker.check_drift();
    assert_eq!(drifts.len(), 1);
    assert_eq!(drifts[0].0, "c");
}

#[test]
fn regression_leak_detector_unknown_free() {
    let mut detector = LeakDetector::new();
    let unknown = next_buffer_id();
    assert!(!detector.track_free(unknown));
}
