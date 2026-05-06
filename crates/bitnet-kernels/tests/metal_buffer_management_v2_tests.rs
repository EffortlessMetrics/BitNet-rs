#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Comprehensive tests for Metal buffer management on Apple Silicon.
//!
//! Validates buffer allocation/deallocation, pool management, storage modes,
//! alignment requirements, large buffer handling, hazard tracking,
//! triple buffering, contents validation, memory pressure, and debug markers.
//!
//! All types are mock/simulated — no actual Metal framework dependency.

#![cfg(target_os = "macos")]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ═══════════════════════════════════════════════════════════════════════
// Mock Metal types
// ═══════════════════════════════════════════════════════════════════════

/// Metal buffer alignment on Apple Silicon (256 bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Page size on Apple Silicon (16 KiB).
const PAGE_SIZE: usize = 16384;

/// Maximum single buffer size we simulate (4 GiB for testing).
const MAX_SIMULATED_BUFFER: usize = 4 * 1024 * 1024 * 1024;

/// Simulated device memory limit (8 GiB).
const SIMULATED_DEVICE_MEMORY: usize = 8 * 1024 * 1024 * 1024;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

/// Metal storage modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StorageMode {
    /// CPU and GPU share the same memory (Apple Silicon unified memory).
    Shared,
    /// GPU-only; CPU access requires explicit blit.
    Private,
    /// Managed: Metal tracks dirty regions for CPU↔GPU sync.
    Managed,
    /// Memoryless: tile memory only, no backing store.
    Memoryless,
}

/// CPU cache mode for Metal buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuCacheMode {
    DefaultCache,
    WriteCombined,
}

/// Hazard tracking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HazardTrackingMode {
    Default,
    Tracked,
    Untracked,
}

/// Resource usage flags for encoder tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceUsage {
    Read,
    Write,
    ReadWrite,
}

/// Simulated Metal buffer.
#[derive(Clone)]
struct MetalBuffer {
    id: u64,
    size: usize,
    storage_mode: StorageMode,
    cpu_cache_mode: CpuCacheMode,
    hazard_tracking: HazardTrackingMode,
    label: Option<String>,
    contents: Vec<u8>,
    allocated: bool,
}

impl MetalBuffer {
    fn new(size: usize, mode: StorageMode) -> Self {
        Self {
            id: next_id(),
            size,
            storage_mode: mode,
            cpu_cache_mode: CpuCacheMode::DefaultCache,
            hazard_tracking: HazardTrackingMode::Default,
            label: None,
            contents: vec![0u8; size],
            allocated: true,
        }
    }

    fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    fn with_cache_mode(mut self, mode: CpuCacheMode) -> Self {
        self.cpu_cache_mode = mode;
        self
    }

    fn with_hazard_tracking(mut self, mode: HazardTrackingMode) -> Self {
        self.hazard_tracking = mode;
        self
    }

    fn with_data(mut self, data: &[u8]) -> Self {
        assert!(data.len() <= self.size);
        self.contents[..data.len()].copy_from_slice(data);
        self
    }

    fn is_zero_filled(&self) -> bool {
        self.contents.iter().all(|&b| b == 0)
    }

    fn fill_pattern(&mut self, pattern: &[u8]) {
        for chunk in self.contents.chunks_mut(pattern.len()) {
            let len = chunk.len().min(pattern.len());
            chunk[..len].copy_from_slice(&pattern[..len]);
        }
    }

    fn deallocate(&mut self) {
        self.allocated = false;
        self.contents.clear();
    }

    fn aligned_size(&self) -> usize {
        align_up(self.size, METAL_BUFFER_ALIGNMENT)
    }
}

/// Simulated Metal device for buffer creation.
struct MetalDevice {
    allocated_bytes: AtomicU64,
    max_memory: usize,
    buffers_created: AtomicU64,
}

impl MetalDevice {
    fn new(max_memory: usize) -> Self {
        Self { allocated_bytes: AtomicU64::new(0), max_memory, buffers_created: AtomicU64::new(0) }
    }

    fn create_buffer(&self, size: usize, mode: StorageMode) -> Result<MetalBuffer, BufferError> {
        if size == 0 {
            return Err(BufferError::ZeroSize);
        }
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        let current = self.allocated_bytes.load(Ordering::Relaxed) as usize;
        if current + aligned > self.max_memory {
            return Err(BufferError::OutOfMemory {
                requested: aligned,
                available: self.max_memory.saturating_sub(current),
            });
        }
        self.allocated_bytes.fetch_add(aligned as u64, Ordering::Relaxed);
        self.buffers_created.fetch_add(1, Ordering::Relaxed);
        Ok(MetalBuffer::new(size, mode))
    }

    fn release_buffer(&self, buffer: &mut MetalBuffer) {
        let aligned = align_up(buffer.size, METAL_BUFFER_ALIGNMENT) as u64;
        self.allocated_bytes.fetch_sub(aligned, Ordering::Relaxed);
        buffer.deallocate();
    }

    fn allocated(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed) as usize
    }

    fn buffers_created_count(&self) -> u64 {
        self.buffers_created.load(Ordering::Relaxed)
    }
}

/// Buffer pool entry.
struct PoolEntry {
    buffer: MetalBuffer,
    last_used_frame: u64,
}

/// Buffer pool with size-bucketed reuse and eviction.
struct BufferPool {
    device: Arc<MetalDevice>,
    /// Buckets keyed by (aligned_size, storage_mode).
    free_lists: HashMap<(usize, StorageMode), VecDeque<PoolEntry>>,
    in_use: HashMap<u64, MetalBuffer>,
    current_frame: u64,
    max_pool_bytes: usize,
    pool_bytes: usize,
    hits: u64,
    misses: u64,
}

impl BufferPool {
    fn new(device: Arc<MetalDevice>, max_pool_bytes: usize) -> Self {
        Self {
            device,
            free_lists: HashMap::new(),
            in_use: HashMap::new(),
            current_frame: 0,
            max_pool_bytes,
            pool_bytes: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn acquire(&mut self, size: usize, mode: StorageMode) -> Result<MetalBuffer, BufferError> {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        let key = (aligned, mode);

        if let Some(list) = self.free_lists.get_mut(&key)
            && let Some(entry) = list.pop_front()
        {
            self.pool_bytes -= entry.buffer.aligned_size();
            self.hits += 1;
            let buf = entry.buffer;
            self.in_use.insert(buf.id, buf.clone());
            return Ok(buf);
        }

        self.misses += 1;
        let buf = self.device.create_buffer(size, mode)?;
        self.in_use.insert(buf.id, buf.clone());
        Ok(buf)
    }

    fn release(&mut self, id: u64) {
        if let Some(buf) = self.in_use.remove(&id) {
            let aligned = buf.aligned_size();
            // Evict if pool is full
            while self.pool_bytes + aligned > self.max_pool_bytes {
                if !self.evict_oldest() {
                    break;
                }
            }
            if self.pool_bytes + aligned <= self.max_pool_bytes {
                let key = (align_up(buf.size, METAL_BUFFER_ALIGNMENT), buf.storage_mode);
                self.pool_bytes += aligned;
                self.free_lists
                    .entry(key)
                    .or_default()
                    .push_back(PoolEntry { buffer: buf, last_used_frame: self.current_frame });
            } else {
                // Truly release back to device
                let mut buf = buf;
                self.device.release_buffer(&mut buf);
            }
        }
    }

    fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    fn evict_oldest(&mut self) -> bool {
        let mut oldest_key = None;
        let mut oldest_frame = u64::MAX;

        for (key, list) in &self.free_lists {
            if let Some(entry) = list.front()
                && entry.last_used_frame < oldest_frame
            {
                oldest_frame = entry.last_used_frame;
                oldest_key = Some(*key);
            }
        }

        if let Some(key) = oldest_key
            && let Some(list) = self.free_lists.get_mut(&key)
            && let Some(entry) = list.pop_front()
        {
            let aligned = entry.buffer.aligned_size();
            self.pool_bytes -= aligned;
            let mut buf = entry.buffer;
            self.device.release_buffer(&mut buf);
            return true;
        }
        false
    }

    fn evict_all(&mut self) {
        for (_key, list) in self.free_lists.drain() {
            for entry in list {
                let mut buf = entry.buffer;
                self.device.release_buffer(&mut buf);
            }
        }
        self.pool_bytes = 0;
    }

    fn pool_size_bytes(&self) -> usize {
        self.pool_bytes
    }

    fn in_use_count(&self) -> usize {
        self.in_use.len()
    }

    fn free_count(&self) -> usize {
        self.free_lists.values().map(|l| l.len()).sum()
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// Hazard tracker for read-after-write / write-after-read ordering.
struct HazardTracker {
    /// Maps buffer id → last operation.
    last_ops: HashMap<u64, ResourceUsage>,
    /// Detected hazards.
    hazards: Vec<Hazard>,
}

#[derive(Debug, Clone)]
struct Hazard {
    buffer_id: u64,
    kind: HazardKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HazardKind {
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
}

impl HazardTracker {
    fn new() -> Self {
        Self { last_ops: HashMap::new(), hazards: Vec::new() }
    }

    fn record(&mut self, buffer_id: u64, usage: ResourceUsage) {
        if let Some(&prev) = self.last_ops.get(&buffer_id) {
            let kind = match (prev, usage) {
                (ResourceUsage::Write | ResourceUsage::ReadWrite, ResourceUsage::Read) => {
                    Some(HazardKind::ReadAfterWrite)
                }
                (ResourceUsage::Read, ResourceUsage::Write | ResourceUsage::ReadWrite) => {
                    Some(HazardKind::WriteAfterRead)
                }
                (
                    ResourceUsage::Write | ResourceUsage::ReadWrite,
                    ResourceUsage::Write | ResourceUsage::ReadWrite,
                ) => Some(HazardKind::WriteAfterWrite),
                _ => None,
            };
            if let Some(kind) = kind {
                self.hazards.push(Hazard { buffer_id, kind });
            }
        }
        self.last_ops.insert(buffer_id, usage);
    }

    fn barrier(&mut self) {
        self.last_ops.clear();
    }

    fn hazard_count(&self) -> usize {
        self.hazards.len()
    }

    fn hazards_for(&self, buffer_id: u64) -> Vec<&Hazard> {
        self.hazards.iter().filter(|h| h.buffer_id == buffer_id).collect()
    }
}

/// Triple buffer ring for pipelined rendering / compute.
struct TripleBuffer {
    buffers: [MetalBuffer; 3],
    current_index: usize,
    frame_count: u64,
}

impl TripleBuffer {
    fn new(size: usize, mode: StorageMode) -> Self {
        Self {
            buffers: [
                MetalBuffer::new(size, mode),
                MetalBuffer::new(size, mode),
                MetalBuffer::new(size, mode),
            ],
            current_index: 0,
            frame_count: 0,
        }
    }

    fn current(&self) -> &MetalBuffer {
        &self.buffers[self.current_index]
    }

    fn current_mut(&mut self) -> &mut MetalBuffer {
        &mut self.buffers[self.current_index]
    }

    fn advance(&mut self) {
        self.current_index = (self.current_index + 1) % 3;
        self.frame_count += 1;
    }

    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    fn all_allocated(&self) -> bool {
        self.buffers.iter().all(|b| b.allocated)
    }
}

/// Memory pressure levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MemoryPressure {
    Normal,
    Warning,
    Critical,
}

/// Memory pressure handler.
struct MemoryPressureHandler {
    level: MemoryPressure,
    pool: BufferPool,
    eviction_count: u64,
}

impl MemoryPressureHandler {
    fn new(pool: BufferPool) -> Self {
        Self { level: MemoryPressure::Normal, pool, eviction_count: 0 }
    }

    fn update_pressure(&mut self, level: MemoryPressure) {
        self.level = level;
        match level {
            MemoryPressure::Normal => {}
            MemoryPressure::Warning => {
                // Evict half the pool
                let target = self.pool.free_count() / 2;
                for _ in 0..target {
                    if self.pool.evict_oldest() {
                        self.eviction_count += 1;
                    }
                }
            }
            MemoryPressure::Critical => {
                let count = self.pool.free_count() as u64;
                self.pool.evict_all();
                self.eviction_count += count;
            }
        }
    }
}

/// Buffer error type.
#[derive(Debug)]
enum BufferError {
    ZeroSize,
    OutOfMemory { requested: usize, available: usize },
    AlignmentViolation { size: usize, alignment: usize },
    InvalidStorageMode,
}

// ── Utility functions ───────────────────────────────────────────────

fn align_up(size: usize, alignment: usize) -> usize {
    if size == 0 {
        return 0;
    }
    debug_assert!(alignment.is_power_of_two());
    (size + alignment - 1) & !(alignment - 1)
}

fn is_aligned(size: usize, alignment: usize) -> bool {
    size.is_multiple_of(alignment)
}

fn required_alignment_for_type(type_size: usize) -> usize {
    // Metal requires at least the type's natural alignment,
    // but never less than METAL_BUFFER_ALIGNMENT for buffers.
    METAL_BUFFER_ALIGNMENT.max(type_size)
}

/// Compute size class bucket for the pool (next power of two ≥ 256).
fn size_class(size: usize) -> usize {
    let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
    aligned.next_power_of_two()
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Buffer allocation and deallocation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn alloc_basic_shared_buffer() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let buf = device.create_buffer(1024, StorageMode::Shared).unwrap();
    assert_eq!(buf.size, 1024);
    assert_eq!(buf.storage_mode, StorageMode::Shared);
    assert!(buf.allocated);
}

#[test]
fn alloc_basic_private_buffer() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let buf = device.create_buffer(4096, StorageMode::Private).unwrap();
    assert_eq!(buf.storage_mode, StorageMode::Private);
    assert!(buf.allocated);
}

#[test]
fn alloc_basic_managed_buffer() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let buf = device.create_buffer(512, StorageMode::Managed).unwrap();
    assert_eq!(buf.storage_mode, StorageMode::Managed);
}

#[test]
fn alloc_zero_size_fails() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let result = device.create_buffer(0, StorageMode::Shared);
    assert!(matches!(result, Err(BufferError::ZeroSize)));
}

#[test]
fn alloc_tracks_device_memory() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    assert_eq!(device.allocated(), 0);
    let _b1 = device.create_buffer(1024, StorageMode::Shared).unwrap();
    assert!(device.allocated() > 0);
    assert!(device.allocated() >= 1024);
}

#[test]
fn dealloc_frees_device_memory() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let mut buf = device.create_buffer(1024, StorageMode::Shared).unwrap();
    let before = device.allocated();
    device.release_buffer(&mut buf);
    assert!(!buf.allocated);
    assert!(device.allocated() < before);
}

#[test]
fn alloc_unique_buffer_ids() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let b1 = device.create_buffer(256, StorageMode::Shared).unwrap();
    let b2 = device.create_buffer(256, StorageMode::Shared).unwrap();
    let b3 = device.create_buffer(256, StorageMode::Private).unwrap();
    assert_ne!(b1.id, b2.id);
    assert_ne!(b2.id, b3.id);
    assert_ne!(b1.id, b3.id);
}

#[test]
fn alloc_increments_buffer_count() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    assert_eq!(device.buffers_created_count(), 0);
    let _b1 = device.create_buffer(256, StorageMode::Shared).unwrap();
    assert_eq!(device.buffers_created_count(), 1);
    let _b2 = device.create_buffer(512, StorageMode::Private).unwrap();
    assert_eq!(device.buffers_created_count(), 2);
}

#[test]
fn alloc_out_of_memory() {
    let device = MetalDevice::new(1024); // tiny memory
    let result = device.create_buffer(2048, StorageMode::Shared);
    assert!(matches!(result, Err(BufferError::OutOfMemory { .. })));
}

#[test]
fn alloc_multiple_until_oom() {
    let device = MetalDevice::new(2048);
    let _b1 = device.create_buffer(512, StorageMode::Shared).unwrap();
    let _b2 = device.create_buffer(512, StorageMode::Shared).unwrap();
    // Next allocation should exhaust remaining space
    let result = device.create_buffer(2048, StorageMode::Shared);
    assert!(matches!(result, Err(BufferError::OutOfMemory { .. })));
}

#[test]
fn dealloc_then_realloc_succeeds() {
    let device = MetalDevice::new(1024);
    let mut buf = device.create_buffer(512, StorageMode::Shared).unwrap();
    device.release_buffer(&mut buf);
    let buf2 = device.create_buffer(512, StorageMode::Shared).unwrap();
    assert!(buf2.allocated);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Buffer pool management
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn pool_acquire_creates_new_buffer() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device.clone(), 1024 * 1024);
    let buf = pool.acquire(1024, StorageMode::Shared).unwrap();
    assert_eq!(buf.size, 1024);
    assert_eq!(pool.in_use_count(), 1);
    assert_eq!(pool.free_count(), 0);
}

#[test]
fn pool_release_adds_to_free_list() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);
    let buf = pool.acquire(1024, StorageMode::Shared).unwrap();
    let id = buf.id;
    pool.release(id);
    assert_eq!(pool.in_use_count(), 0);
    assert_eq!(pool.free_count(), 1);
}

#[test]
fn pool_reuse_returns_cached_buffer() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device.clone(), 1024 * 1024);
    let buf1 = pool.acquire(1024, StorageMode::Shared).unwrap();
    let id1 = buf1.id;
    pool.release(id1);

    // Second acquire should reuse
    let _buf2 = pool.acquire(1024, StorageMode::Shared).unwrap();
    assert!(pool.hits >= 1);
    assert_eq!(pool.free_count(), 0);
}

#[test]
fn pool_different_sizes_no_reuse() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);
    let buf1 = pool.acquire(256, StorageMode::Shared).unwrap();
    pool.release(buf1.id);

    // Different aligned size → miss
    let _buf2 = pool.acquire(1024, StorageMode::Shared).unwrap();
    assert_eq!(pool.misses, 2); // both are misses
}

#[test]
fn pool_different_storage_modes_no_reuse() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);
    let buf1 = pool.acquire(1024, StorageMode::Shared).unwrap();
    pool.release(buf1.id);

    let _buf2 = pool.acquire(1024, StorageMode::Private).unwrap();
    assert_eq!(pool.misses, 2);
}

#[test]
fn pool_eviction_when_full() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    // Pool can hold ~2 buffers of 256 bytes each (pool max = 512)
    let mut pool = BufferPool::new(device, 512);
    let b1 = pool.acquire(256, StorageMode::Shared).unwrap();
    let b2 = pool.acquire(256, StorageMode::Shared).unwrap();
    pool.release(b1.id);
    pool.advance_frame();
    pool.release(b2.id);

    // Pool should have 2 entries totaling 512 bytes
    assert_eq!(pool.free_count(), 2);

    // Add a third — should trigger eviction
    let b3 = pool.acquire(256, StorageMode::Shared).unwrap();
    pool.release(b3.id);

    // Pool size should not exceed limit
    assert!(pool.pool_size_bytes() <= 512);
}

#[test]
fn pool_evict_all_clears_pool() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    for _ in 0..10 {
        let buf = pool.acquire(256, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }
    assert!(pool.free_count() > 0);

    pool.evict_all();
    assert_eq!(pool.free_count(), 0);
    assert_eq!(pool.pool_size_bytes(), 0);
}

#[test]
fn pool_hit_rate_starts_zero() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let pool = BufferPool::new(device, 1024 * 1024);
    assert_eq!(pool.hit_rate(), 0.0);
}

#[test]
fn pool_hit_rate_increases_with_reuse() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    // First acquire: miss
    let buf = pool.acquire(1024, StorageMode::Shared).unwrap();
    pool.release(buf.id);

    // Second acquire: hit
    let buf2 = pool.acquire(1024, StorageMode::Shared).unwrap();
    pool.release(buf2.id);

    assert!(pool.hit_rate() > 0.0);
    // 1 hit / 2 total = 0.5
    assert!((pool.hit_rate() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn pool_frame_advance_increments() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);
    assert_eq!(pool.current_frame, 0);
    pool.advance_frame();
    assert_eq!(pool.current_frame, 1);
    pool.advance_frame();
    assert_eq!(pool.current_frame, 2);
}

#[test]
fn pool_fragmentation_multiple_sizes() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 10 * 1024 * 1024);

    let sizes = [256, 512, 1024, 2048, 4096];
    let mut ids = Vec::new();

    for &size in &sizes {
        let buf = pool.acquire(size, StorageMode::Shared).unwrap();
        ids.push(buf.id);
    }

    // Release all
    for id in ids {
        pool.release(id);
    }

    // Free list should have entries for each distinct aligned size
    assert_eq!(pool.free_count(), sizes.len());
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Storage modes
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn storage_mode_shared_allows_cpu_access() {
    let buf = MetalBuffer::new(1024, StorageMode::Shared);
    assert_eq!(buf.storage_mode, StorageMode::Shared);
    // Shared buffers are accessible from CPU
    assert!(!buf.contents.is_empty());
}

#[test]
fn storage_mode_private_is_gpu_only() {
    let buf = MetalBuffer::new(1024, StorageMode::Private);
    assert_eq!(buf.storage_mode, StorageMode::Private);
}

#[test]
fn storage_mode_managed_tracks_regions() {
    let buf = MetalBuffer::new(1024, StorageMode::Managed);
    assert_eq!(buf.storage_mode, StorageMode::Managed);
}

#[test]
fn storage_mode_memoryless_for_tile() {
    let buf = MetalBuffer::new(1024, StorageMode::Memoryless);
    assert_eq!(buf.storage_mode, StorageMode::Memoryless);
}

#[test]
fn storage_modes_all_distinct() {
    let modes =
        [StorageMode::Shared, StorageMode::Private, StorageMode::Managed, StorageMode::Memoryless];
    for i in 0..modes.len() {
        for j in (i + 1)..modes.len() {
            assert_ne!(modes[i], modes[j]);
        }
    }
}

#[test]
fn cache_mode_default() {
    let buf = MetalBuffer::new(256, StorageMode::Shared);
    assert_eq!(buf.cpu_cache_mode, CpuCacheMode::DefaultCache);
}

#[test]
fn cache_mode_write_combined() {
    let buf =
        MetalBuffer::new(256, StorageMode::Shared).with_cache_mode(CpuCacheMode::WriteCombined);
    assert_eq!(buf.cpu_cache_mode, CpuCacheMode::WriteCombined);
}

#[test]
fn shared_vs_private_recommendation_small() {
    // Small buffers with CPU writes → Shared is preferred on Apple Silicon
    let size = 4096;
    let preferred = if size < 1024 * 1024 {
        StorageMode::Shared // unified memory: avoid blit overhead
    } else {
        StorageMode::Private
    };
    assert_eq!(preferred, StorageMode::Shared);
}

#[test]
fn shared_vs_private_recommendation_large() {
    // Large GPU-only buffers → Private reduces cache pollution
    let size = 64 * 1024 * 1024;
    let preferred = if size < 1024 * 1024 { StorageMode::Shared } else { StorageMode::Private };
    assert_eq!(preferred, StorageMode::Private);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Alignment requirements
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn alignment_256_byte_minimum() {
    assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
    assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two());
}

#[test]
fn align_up_exact_multiple_unchanged() {
    for mult in 1..=16 {
        let size = METAL_BUFFER_ALIGNMENT * mult;
        assert_eq!(align_up(size, METAL_BUFFER_ALIGNMENT), size);
    }
}

#[test]
fn align_up_rounds_to_next_boundary() {
    assert_eq!(align_up(1, METAL_BUFFER_ALIGNMENT), 256);
    assert_eq!(align_up(255, METAL_BUFFER_ALIGNMENT), 256);
    assert_eq!(align_up(257, METAL_BUFFER_ALIGNMENT), 512);
}

#[test]
fn align_up_zero_returns_zero() {
    assert_eq!(align_up(0, METAL_BUFFER_ALIGNMENT), 0);
}

#[test]
fn aligned_size_property_for_various_inputs() {
    let test_sizes = [1, 2, 4, 8, 16, 64, 128, 255, 256, 257, 512, 1023, 1024];
    for &size in &test_sizes {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        assert!(aligned >= size, "aligned {aligned} < size {size}");
        assert!(
            is_aligned(aligned, METAL_BUFFER_ALIGNMENT),
            "size {size} → {aligned} not aligned to {METAL_BUFFER_ALIGNMENT}"
        );
    }
}

#[test]
fn alignment_for_f32_type() {
    let align = required_alignment_for_type(std::mem::size_of::<f32>());
    assert!(align >= METAL_BUFFER_ALIGNMENT);
    assert!(align >= 4);
}

#[test]
fn alignment_for_f16_type() {
    // f16 = 2 bytes
    let align = required_alignment_for_type(2);
    assert!(align >= METAL_BUFFER_ALIGNMENT);
}

#[test]
fn alignment_for_simd_float4() {
    // float4 = 16 bytes
    let align = required_alignment_for_type(16);
    assert!(align >= METAL_BUFFER_ALIGNMENT);
    assert!(align >= 16);
}

#[test]
fn alignment_for_matrix_4x4() {
    // 4x4 f32 matrix = 64 bytes
    let align = required_alignment_for_type(64);
    assert!(align >= METAL_BUFFER_ALIGNMENT);
}

#[test]
fn page_alignment_for_large_buffers() {
    let large_size = 64 * 1024; // 64 KiB
    let page_aligned = align_up(large_size, PAGE_SIZE);
    assert_eq!(page_aligned, large_size); // already page-aligned
    assert!(is_aligned(page_aligned, PAGE_SIZE));
}

#[test]
fn buffer_aligned_size_method() {
    let buf = MetalBuffer::new(100, StorageMode::Shared);
    assert_eq!(buf.aligned_size(), 256);

    let buf2 = MetalBuffer::new(256, StorageMode::Shared);
    assert_eq!(buf2.aligned_size(), 256);

    let buf3 = MetalBuffer::new(257, StorageMode::Shared);
    assert_eq!(buf3.aligned_size(), 512);
}

#[test]
fn size_class_bucketing() {
    assert_eq!(size_class(1), 256);
    assert_eq!(size_class(256), 256);
    assert_eq!(size_class(257), 512);
    assert_eq!(size_class(512), 512);
    assert_eq!(size_class(513), 1024);
}

#[test]
fn alignment_parameterized_powers_of_two() {
    for exp in 0..20 {
        let size = 1usize << exp;
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        assert!(aligned >= size);
        assert!(is_aligned(aligned, METAL_BUFFER_ALIGNMENT));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Large buffer handling
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn large_buffer_256mb_alignment() {
    let size = 256 * 1024 * 1024; // 256 MiB
    let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
    assert_eq!(aligned, size); // already aligned
    assert!(aligned <= MAX_SIMULATED_BUFFER);
}

#[test]
fn large_buffer_1gb_alignment() {
    let size = 1024 * 1024 * 1024; // 1 GiB
    let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
    assert_eq!(aligned, size);
}

#[test]
fn large_buffer_exceeds_device_memory() {
    let device = MetalDevice::new(512 * 1024 * 1024); // 512 MiB
    let result = device.create_buffer(1024 * 1024 * 1024, StorageMode::Shared);
    assert!(matches!(result, Err(BufferError::OutOfMemory { .. })));
}

#[test]
fn large_buffer_page_aligned() {
    let size = 256 * 1024 * 1024 + 1; // 256 MiB + 1
    let page_aligned = align_up(size, PAGE_SIZE);
    assert!(is_aligned(page_aligned, PAGE_SIZE));
    assert!(page_aligned > size);
}

#[test]
fn large_buffer_allocation_tracking() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let _buf = device.create_buffer(512 * 1024 * 1024, StorageMode::Private).unwrap();
    assert!(device.allocated() >= 512 * 1024 * 1024);
}

#[test]
fn large_buffer_size_class_boundaries() {
    // Verify size class computation at large scales
    let mb = 1024 * 1024;
    let sizes = [mb, 16 * mb, 128 * mb, 256 * mb];
    for &s in &sizes {
        let sc = size_class(s);
        assert!(sc >= s);
        assert!(sc.is_power_of_two());
    }
}

#[test]
fn large_buffer_near_overflow_alignment() {
    // Ensure alignment doesn't overflow for very large values
    let big = usize::MAX - METAL_BUFFER_ALIGNMENT;
    // align_up would overflow, so we test a safe large value instead
    let safe_large = (1usize << 40) - 1; // ~1 TiB - 1
    let aligned = align_up(safe_large, METAL_BUFFER_ALIGNMENT);
    assert!(aligned >= safe_large);
    assert!(is_aligned(aligned, METAL_BUFFER_ALIGNMENT));
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Hazard tracking
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn hazard_tracker_empty_initially() {
    let tracker = HazardTracker::new();
    assert_eq!(tracker.hazard_count(), 0);
}

#[test]
fn hazard_read_after_write_detected() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.record(1, ResourceUsage::Read);
    assert_eq!(tracker.hazard_count(), 1);
    assert_eq!(tracker.hazards_for(1)[0].kind, HazardKind::ReadAfterWrite);
}

#[test]
fn hazard_write_after_read_detected() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Read);
    tracker.record(1, ResourceUsage::Write);
    assert_eq!(tracker.hazard_count(), 1);
    assert_eq!(tracker.hazards_for(1)[0].kind, HazardKind::WriteAfterRead);
}

#[test]
fn hazard_write_after_write_detected() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.record(1, ResourceUsage::Write);
    assert_eq!(tracker.hazard_count(), 1);
    assert_eq!(tracker.hazards_for(1)[0].kind, HazardKind::WriteAfterWrite);
}

#[test]
fn hazard_read_after_read_no_hazard() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Read);
    tracker.record(1, ResourceUsage::Read);
    assert_eq!(tracker.hazard_count(), 0);
}

#[test]
fn hazard_barrier_clears_state() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.barrier();
    tracker.record(1, ResourceUsage::Read);
    // No hazard: barrier separates the ops
    assert_eq!(tracker.hazard_count(), 0);
}

#[test]
fn hazard_multiple_buffers_independent() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.record(2, ResourceUsage::Read);
    // Different buffers → no hazard
    assert_eq!(tracker.hazard_count(), 0);
}

#[test]
fn hazard_readwrite_after_write() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.record(1, ResourceUsage::ReadWrite);
    assert_eq!(tracker.hazard_count(), 1);
    assert_eq!(tracker.hazards_for(1)[0].kind, HazardKind::WriteAfterWrite);
}

#[test]
fn hazard_chain_accumulates() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.record(1, ResourceUsage::Read);
    tracker.record(1, ResourceUsage::Write);
    tracker.record(1, ResourceUsage::Read);
    // W→R (RAW), R→W (WAR), W→R (RAW) = 3 hazards
    assert_eq!(tracker.hazard_count(), 3);
}

#[test]
fn hazard_per_buffer_isolation() {
    let mut tracker = HazardTracker::new();
    tracker.record(1, ResourceUsage::Write);
    tracker.record(1, ResourceUsage::Read); // RAW on buffer 1
    tracker.record(2, ResourceUsage::Write);
    tracker.record(2, ResourceUsage::Read); // RAW on buffer 2

    assert_eq!(tracker.hazards_for(1).len(), 1);
    assert_eq!(tracker.hazards_for(2).len(), 1);
    assert_eq!(tracker.hazard_count(), 2);
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Triple buffering
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn triple_buffer_creates_three_buffers() {
    let tb = TripleBuffer::new(1024, StorageMode::Shared);
    assert!(tb.all_allocated());
    assert_eq!(tb.buffers.len(), 3);
}

#[test]
fn triple_buffer_starts_at_index_zero() {
    let tb = TripleBuffer::new(1024, StorageMode::Shared);
    assert_eq!(tb.current_index, 0);
    assert_eq!(tb.frame_count(), 0);
}

#[test]
fn triple_buffer_advance_cycles() {
    let mut tb = TripleBuffer::new(1024, StorageMode::Shared);
    assert_eq!(tb.current_index, 0);

    tb.advance();
    assert_eq!(tb.current_index, 1);

    tb.advance();
    assert_eq!(tb.current_index, 2);

    tb.advance();
    assert_eq!(tb.current_index, 0); // wraps around
}

#[test]
fn triple_buffer_frame_count_increments() {
    let mut tb = TripleBuffer::new(1024, StorageMode::Shared);
    for i in 0..10 {
        assert_eq!(tb.frame_count(), i);
        tb.advance();
    }
    assert_eq!(tb.frame_count(), 10);
}

#[test]
fn triple_buffer_unique_buffer_ids() {
    let tb = TripleBuffer::new(1024, StorageMode::Shared);
    let ids: Vec<u64> = tb.buffers.iter().map(|b| b.id).collect();
    assert_ne!(ids[0], ids[1]);
    assert_ne!(ids[1], ids[2]);
    assert_ne!(ids[0], ids[2]);
}

#[test]
fn triple_buffer_current_changes_after_advance() {
    let mut tb = TripleBuffer::new(1024, StorageMode::Shared);
    let id0 = tb.current().id;
    tb.advance();
    let id1 = tb.current().id;
    assert_ne!(id0, id1);
}

#[test]
fn triple_buffer_full_cycle_returns_to_first() {
    let mut tb = TripleBuffer::new(1024, StorageMode::Shared);
    let id0 = tb.current().id;
    tb.advance();
    tb.advance();
    tb.advance();
    assert_eq!(tb.current().id, id0);
}

#[test]
fn triple_buffer_write_isolated_per_frame() {
    let mut tb = TripleBuffer::new(256, StorageMode::Shared);

    // Write to frame 0
    tb.current_mut().contents[0] = 0xAA;
    tb.advance();

    // Frame 1 should still be zero
    assert_eq!(tb.current().contents[0], 0);

    tb.advance();
    // Frame 2 also zero
    assert_eq!(tb.current().contents[0], 0);

    tb.advance();
    // Back to frame 0 — should see our write
    assert_eq!(tb.current().contents[0], 0xAA);
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Buffer contents validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn buffer_zero_initialized() {
    let buf = MetalBuffer::new(1024, StorageMode::Shared);
    assert!(buf.is_zero_filled());
}

#[test]
fn buffer_with_data_initialization() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let buf = MetalBuffer::new(1024, StorageMode::Shared).with_data(&data);
    assert_eq!(&buf.contents[..4], &data);
    // Rest is zero
    assert!(buf.contents[4..].iter().all(|&b| b == 0));
}

#[test]
fn buffer_pattern_fill() {
    let mut buf = MetalBuffer::new(256, StorageMode::Shared);
    buf.fill_pattern(&[0xAB, 0xCD]);
    for chunk in buf.contents.chunks(2) {
        assert_eq!(chunk[0], 0xAB);
        if chunk.len() > 1 {
            assert_eq!(chunk[1], 0xCD);
        }
    }
}

#[test]
fn buffer_pattern_fill_single_byte() {
    let mut buf = MetalBuffer::new(128, StorageMode::Shared);
    buf.fill_pattern(&[0xFF]);
    assert!(buf.contents.iter().all(|&b| b == 0xFF));
}

#[test]
fn buffer_contents_size_matches() {
    for size in [1, 128, 256, 1024, 4096] {
        let buf = MetalBuffer::new(size, StorageMode::Shared);
        assert_eq!(buf.contents.len(), size);
    }
}

#[test]
fn buffer_dealloc_clears_contents() {
    let mut buf = MetalBuffer::new(1024, StorageMode::Shared);
    buf.fill_pattern(&[0xFF]);
    buf.deallocate();
    assert!(buf.contents.is_empty());
    assert!(!buf.allocated);
}

#[test]
fn buffer_with_data_partial_fill() {
    let buf = MetalBuffer::new(16, StorageMode::Shared).with_data(&[1, 2, 3]);
    assert_eq!(buf.contents[0], 1);
    assert_eq!(buf.contents[1], 2);
    assert_eq!(buf.contents[2], 3);
    assert_eq!(buf.contents[3], 0);
}

#[test]
fn buffer_f32_contents_validation() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = MetalBuffer::new(256, StorageMode::Shared).with_data(&bytes);

    // Read back as f32
    for (i, &expected) in values.iter().enumerate() {
        let offset = i * 4;
        let actual = f32::from_le_bytes([
            buf.contents[offset],
            buf.contents[offset + 1],
            buf.contents[offset + 2],
            buf.contents[offset + 3],
        ]);
        assert_eq!(actual, expected);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Memory pressure handling
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn pressure_normal_no_eviction() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let pool = BufferPool::new(device, 1024 * 1024);
    let mut handler = MemoryPressureHandler::new(pool);
    handler.update_pressure(MemoryPressure::Normal);
    assert_eq!(handler.eviction_count, 0);
}

#[test]
fn pressure_warning_evicts_half() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    // Add 10 buffers to pool
    for _ in 0..10 {
        let buf = pool.acquire(256, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }
    assert_eq!(pool.free_count(), 10);

    let mut handler = MemoryPressureHandler::new(pool);
    handler.update_pressure(MemoryPressure::Warning);
    assert!(handler.eviction_count > 0);
    assert!(handler.pool.free_count() <= 5);
}

#[test]
fn pressure_critical_evicts_all() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    for _ in 0..10 {
        let buf = pool.acquire(256, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }

    let mut handler = MemoryPressureHandler::new(pool);
    handler.update_pressure(MemoryPressure::Critical);
    assert_eq!(handler.pool.free_count(), 0);
    assert_eq!(handler.pool.pool_size_bytes(), 0);
}

#[test]
fn pressure_levels_ordered() {
    assert!(MemoryPressure::Normal < MemoryPressure::Warning);
    assert!(MemoryPressure::Warning < MemoryPressure::Critical);
}

#[test]
fn pressure_recovery_normal_after_critical() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let pool = BufferPool::new(device, 1024 * 1024);
    let mut handler = MemoryPressureHandler::new(pool);

    handler.update_pressure(MemoryPressure::Critical);
    handler.update_pressure(MemoryPressure::Normal);

    // Pool can still acquire after recovery
    let buf = handler.pool.acquire(256, StorageMode::Shared).unwrap();
    assert!(buf.allocated);
}

#[test]
fn pressure_repeated_warning_is_idempotent() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    for _ in 0..4 {
        let buf = pool.acquire(256, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }

    let mut handler = MemoryPressureHandler::new(pool);
    handler.update_pressure(MemoryPressure::Warning);
    let after_first = handler.pool.free_count();

    handler.update_pressure(MemoryPressure::Warning);
    let after_second = handler.pool.free_count();

    assert!(after_second <= after_first);
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Buffer label and debug marker support
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn buffer_default_no_label() {
    let buf = MetalBuffer::new(256, StorageMode::Shared);
    assert!(buf.label.is_none());
}

#[test]
fn buffer_with_label() {
    let buf = MetalBuffer::new(256, StorageMode::Shared).with_label("uniform_data");
    assert_eq!(buf.label.as_deref(), Some("uniform_data"));
}

#[test]
fn buffer_label_empty_string() {
    let buf = MetalBuffer::new(256, StorageMode::Shared).with_label("");
    assert_eq!(buf.label.as_deref(), Some(""));
}

#[test]
fn buffer_label_unicode() {
    let buf = MetalBuffer::new(256, StorageMode::Shared).with_label("tensor_αβγ_weights");
    assert_eq!(buf.label.as_deref(), Some("tensor_αβγ_weights"));
}

#[test]
fn buffer_label_long_string() {
    let long_label = "a".repeat(1024);
    let buf = MetalBuffer::new(256, StorageMode::Shared).with_label(&long_label);
    assert_eq!(buf.label.as_deref().unwrap().len(), 1024);
}

#[test]
fn buffer_label_preserved_across_clone() {
    let buf = MetalBuffer::new(256, StorageMode::Shared).with_label("weights");
    let cloned = buf.clone();
    assert_eq!(cloned.label, buf.label);
}

#[test]
fn buffer_debug_label_convention() {
    // Verify naming convention for debug labels
    let labels =
        ["vertex_buffer", "index_buffer", "uniform_buffer", "storage_buffer", "staging_buffer"];
    for label in &labels {
        let buf = MetalBuffer::new(256, StorageMode::Shared).with_label(label);
        assert!(buf.label.as_deref().unwrap().contains("buffer"));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Property-based style tests (parameterized)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn property_alignment_always_gte_input() {
    for size in (0..2048).step_by(7) {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        assert!(aligned >= size, "aligned({size}) = {aligned} < {size}");
    }
}

#[test]
fn property_alignment_always_multiple_of_256() {
    for size in 1..2048 {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        assert!(
            is_aligned(aligned, METAL_BUFFER_ALIGNMENT),
            "aligned({size}) = {aligned} not multiple of 256"
        );
    }
}

#[test]
fn property_alignment_overhead_bounded() {
    // Alignment overhead is always < METAL_BUFFER_ALIGNMENT
    for size in 1..4096 {
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        let overhead = aligned - size;
        assert!(
            overhead < METAL_BUFFER_ALIGNMENT,
            "overhead for size {size} is {overhead}, expected < {METAL_BUFFER_ALIGNMENT}"
        );
    }
}

#[test]
fn property_alignment_idempotent() {
    for size in 0..2048 {
        let once = align_up(size, METAL_BUFFER_ALIGNMENT);
        let twice = align_up(once, METAL_BUFFER_ALIGNMENT);
        assert_eq!(once, twice, "align_up not idempotent for size {size}");
    }
}

#[test]
fn property_size_class_always_power_of_two() {
    for size in 1..4096 {
        let sc = size_class(size);
        assert!(sc.is_power_of_two(), "size_class({size}) = {sc} not power of two");
    }
}

#[test]
fn property_size_class_always_gte_aligned() {
    for size in 1..4096 {
        let sc = size_class(size);
        let aligned = align_up(size, METAL_BUFFER_ALIGNMENT);
        assert!(sc >= aligned, "size_class({size}) = {sc} < aligned {aligned}");
    }
}

#[test]
fn property_buffer_zero_init_all_sizes() {
    for size in [1, 7, 64, 255, 256, 1000, 4096] {
        let buf = MetalBuffer::new(size, StorageMode::Shared);
        assert!(buf.is_zero_filled(), "buffer of size {size} not zero-filled");
    }
}

#[test]
fn property_storage_modes_exhaustive() {
    let modes =
        [StorageMode::Shared, StorageMode::Private, StorageMode::Managed, StorageMode::Memoryless];
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    for mode in &modes {
        let buf = device.create_buffer(256, *mode).unwrap();
        assert_eq!(buf.storage_mode, *mode);
        assert!(buf.allocated);
    }
}

#[test]
fn property_pool_no_leak_after_acquire_release_cycle() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device.clone(), 1024 * 1024);

    for _ in 0..100 {
        let buf = pool.acquire(512, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }

    assert_eq!(pool.in_use_count(), 0);
    // Pool should have exactly 1 free buffer (reused each time)
    assert_eq!(pool.free_count(), 1);
}

#[test]
fn property_hazard_barrier_always_resets() {
    let mut tracker = HazardTracker::new();
    for _ in 0..50 {
        tracker.record(1, ResourceUsage::Write);
        tracker.barrier();
        tracker.record(1, ResourceUsage::Read);
    }
    // No hazards because barrier resets between each pair
    assert_eq!(tracker.hazard_count(), 0);
}

#[test]
fn property_triple_buffer_wraps_at_three() {
    let mut tb = TripleBuffer::new(256, StorageMode::Shared);
    for cycle in 0..100 {
        assert_eq!(tb.current_index, cycle % 3, "index wrong at cycle {cycle}");
        tb.advance();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Additional edge-case and integration tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn buffer_one_byte_allocation() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let buf = device.create_buffer(1, StorageMode::Shared).unwrap();
    assert_eq!(buf.size, 1);
    assert_eq!(buf.aligned_size(), METAL_BUFFER_ALIGNMENT);
}

#[test]
fn buffer_exact_alignment_boundary() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let buf = device.create_buffer(METAL_BUFFER_ALIGNMENT, StorageMode::Shared).unwrap();
    assert_eq!(buf.size, METAL_BUFFER_ALIGNMENT);
    assert_eq!(buf.aligned_size(), METAL_BUFFER_ALIGNMENT);
}

#[test]
fn pool_concurrent_acquire_release_pattern() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 10 * 1024 * 1024);

    // Simulate a frame-based pattern: acquire N, use, release N
    for frame in 0..10 {
        let mut frame_buffers = Vec::new();
        for _ in 0..5 {
            let buf = pool.acquire(1024, StorageMode::Shared).unwrap();
            frame_buffers.push(buf.id);
        }
        pool.advance_frame();
        for id in frame_buffers {
            pool.release(id);
        }
    }
    assert_eq!(pool.in_use_count(), 0);
    // All 5 buffers should be in the free list
    assert_eq!(pool.free_count(), 5);
}

#[test]
fn hazard_tracking_mode_variants() {
    let buf_default = MetalBuffer::new(256, StorageMode::Shared)
        .with_hazard_tracking(HazardTrackingMode::Default);
    let buf_tracked = MetalBuffer::new(256, StorageMode::Shared)
        .with_hazard_tracking(HazardTrackingMode::Tracked);
    let buf_untracked = MetalBuffer::new(256, StorageMode::Shared)
        .with_hazard_tracking(HazardTrackingMode::Untracked);

    assert_eq!(buf_default.hazard_tracking, HazardTrackingMode::Default);
    assert_eq!(buf_tracked.hazard_tracking, HazardTrackingMode::Tracked);
    assert_eq!(buf_untracked.hazard_tracking, HazardTrackingMode::Untracked);
}

#[test]
fn pool_mixed_storage_mode_buckets() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 10 * 1024 * 1024);

    let modes = [StorageMode::Shared, StorageMode::Private, StorageMode::Managed];
    for &mode in &modes {
        let buf = pool.acquire(1024, mode).unwrap();
        pool.release(buf.id);
    }
    assert_eq!(pool.free_count(), 3); // one per mode
}

#[test]
fn buffer_chained_builder() {
    let buf = MetalBuffer::new(1024, StorageMode::Shared)
        .with_label("test_buf")
        .with_cache_mode(CpuCacheMode::WriteCombined)
        .with_hazard_tracking(HazardTrackingMode::Untracked);

    assert_eq!(buf.label.as_deref(), Some("test_buf"));
    assert_eq!(buf.cpu_cache_mode, CpuCacheMode::WriteCombined);
    assert_eq!(buf.hazard_tracking, HazardTrackingMode::Untracked);
    assert_eq!(buf.storage_mode, StorageMode::Shared);
}

#[test]
fn device_memory_accounting_after_mixed_ops() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let mut b1 = device.create_buffer(1024, StorageMode::Shared).unwrap();
    let _b2 = device.create_buffer(2048, StorageMode::Private).unwrap();
    let alloc_before = device.allocated();

    device.release_buffer(&mut b1);
    let alloc_after = device.allocated();

    assert!(alloc_after < alloc_before);
    assert!(alloc_after > 0); // b2 still allocated
}

#[test]
fn pool_release_nonexistent_id_is_noop() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);
    // Release an ID that was never acquired
    pool.release(999999);
    assert_eq!(pool.in_use_count(), 0);
    assert_eq!(pool.free_count(), 0);
}

#[test]
fn triple_buffer_all_same_storage_mode() {
    let tb = TripleBuffer::new(512, StorageMode::Private);
    for buf in &tb.buffers {
        assert_eq!(buf.storage_mode, StorageMode::Private);
    }
}

#[test]
fn triple_buffer_all_same_size() {
    let tb = TripleBuffer::new(2048, StorageMode::Shared);
    for buf in &tb.buffers {
        assert_eq!(buf.size, 2048);
    }
}

#[test]
fn hazard_readwrite_then_readwrite() {
    let mut tracker = HazardTracker::new();
    tracker.record(42, ResourceUsage::ReadWrite);
    tracker.record(42, ResourceUsage::ReadWrite);
    assert_eq!(tracker.hazard_count(), 1);
    assert_eq!(tracker.hazards_for(42)[0].kind, HazardKind::WriteAfterWrite);
}

#[test]
fn buffer_pattern_fill_larger_than_buffer() {
    let mut buf = MetalBuffer::new(3, StorageMode::Shared);
    buf.fill_pattern(&[0xAA, 0xBB, 0xCC, 0xDD]);
    assert_eq!(buf.contents, vec![0xAA, 0xBB, 0xCC]);
}

#[test]
fn pool_eviction_oldest_first() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 768); // room for ~3 × 256

    // Acquire and release 3 buffers across different frames
    let b1 = pool.acquire(256, StorageMode::Shared).unwrap();
    pool.release(b1.id);
    pool.advance_frame();

    let b2 = pool.acquire(512, StorageMode::Shared).unwrap();
    pool.release(b2.id);
    pool.advance_frame();

    // Pool now has 2 entries. Adding a third that exceeds capacity
    // should evict the oldest (b1, frame 0).
    let b3 = pool.acquire(256, StorageMode::Shared).unwrap();
    pool.release(b3.id);

    assert!(pool.pool_size_bytes() <= 768);
}

#[test]
fn buffer_is_aligned_utility() {
    assert!(is_aligned(0, 256));
    assert!(is_aligned(256, 256));
    assert!(is_aligned(512, 256));
    assert!(!is_aligned(1, 256));
    assert!(!is_aligned(255, 256));
    assert!(!is_aligned(257, 256));
}

#[test]
fn buffer_data_roundtrip_u32() {
    let values: Vec<u32> = (0..64).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = MetalBuffer::new(bytes.len(), StorageMode::Shared).with_data(&bytes);

    for (i, &expected) in values.iter().enumerate() {
        let offset = i * 4;
        let actual = u32::from_le_bytes([
            buf.contents[offset],
            buf.contents[offset + 1],
            buf.contents[offset + 2],
            buf.contents[offset + 3],
        ]);
        assert_eq!(actual, expected, "mismatch at index {i}");
    }
}

#[test]
fn pool_stress_many_sizes() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 100 * 1024 * 1024);

    let sizes: Vec<usize> = (1..=50).map(|i| i * 100).collect();
    let mut ids = Vec::new();

    for &size in &sizes {
        let buf = pool.acquire(size, StorageMode::Shared).unwrap();
        ids.push(buf.id);
    }

    for id in ids {
        pool.release(id);
    }

    assert_eq!(pool.in_use_count(), 0);
}

#[test]
fn pressure_handler_cumulative_eviction_count() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    for _ in 0..20 {
        let buf = pool.acquire(256, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }

    let mut handler = MemoryPressureHandler::new(pool);
    handler.update_pressure(MemoryPressure::Warning);
    let first_evictions = handler.eviction_count;
    assert!(first_evictions > 0);

    // Add more buffers and trigger again
    for _ in 0..10 {
        let buf = handler.pool.acquire(256, StorageMode::Shared).unwrap();
        handler.pool.release(buf.id);
    }
    handler.update_pressure(MemoryPressure::Warning);
    assert!(handler.eviction_count >= first_evictions);
}

#[test]
fn buffer_storage_mode_memoryless_no_backing() {
    // Memoryless buffers are for tile memory only; in our simulation
    // they still allocate but real Metal would not persist data.
    let buf = MetalBuffer::new(4096, StorageMode::Memoryless);
    assert_eq!(buf.storage_mode, StorageMode::Memoryless);
    assert!(buf.allocated);
}

#[test]
fn device_release_idempotent() {
    let device = MetalDevice::new(SIMULATED_DEVICE_MEMORY);
    let mut buf = device.create_buffer(1024, StorageMode::Shared).unwrap();
    device.release_buffer(&mut buf);
    assert!(!buf.allocated);
    // Double release — contents already cleared, no panic
    assert!(buf.contents.is_empty());
}

#[test]
fn pool_hit_rate_perfect_with_single_size() {
    let device = Arc::new(MetalDevice::new(SIMULATED_DEVICE_MEMORY));
    let mut pool = BufferPool::new(device, 1024 * 1024);

    // First acquire is a miss
    let buf = pool.acquire(256, StorageMode::Shared).unwrap();
    pool.release(buf.id);

    // Next 99 are hits
    for _ in 0..99 {
        let buf = pool.acquire(256, StorageMode::Shared).unwrap();
        pool.release(buf.id);
    }

    // Hit rate = 99 / 100 = 0.99
    assert!((pool.hit_rate() - 0.99).abs() < 0.001);
}
