//! Metal buffer management validation tests for Apple Silicon.
//!
//! Validates Metal GPU buffer allocation, deallocation, transfer
//! patterns, memory pressure handling, buffer aliasing, and argument
//! buffer layouts. All types are pure Rust mocks — no Metal SDK
//! dependency.

#![cfg(feature = "cpu")]

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

const ALIGN_16: usize = 16;
const ALIGN_256: usize = 256;
const ALIGN_4096: usize = 4096;
const PAGE_SIZE: usize = 16384;
const SIMULATED_BUDGET: usize = 512 * 1024 * 1024; // 512 MiB

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ═══════════════════════════════════════════════════════════════════
// Mock types
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StorageMode {
    Shared,
    Private,
    Managed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PurgeState {
    NonVolatile,
    Volatile,
    Purged,
}

#[derive(Debug, Clone)]
struct MtlBuffer {
    id: u64,
    size: usize,
    aligned_size: usize,
    storage_mode: StorageMode,
    data: Vec<u8>,
    purge_state: PurgeState,
    label: Option<String>,
}

impl MtlBuffer {
    fn new(size: usize, mode: StorageMode, alignment: usize) -> Self {
        let aligned = align_up(size, alignment);
        Self {
            id: next_id(),
            size,
            aligned_size: aligned,
            storage_mode: mode,
            data: vec![0u8; aligned],
            purge_state: PurgeState::NonVolatile,
            label: None,
        }
    }

    fn write(&mut self, offset: usize, src: &[u8]) {
        self.data[offset..offset + src.len()].copy_from_slice(src);
    }

    fn read(&self, offset: usize, len: usize) -> &[u8] {
        &self.data[offset..offset + len]
    }
}

#[derive(Debug)]
struct MtlDevice {
    allocated: usize,
    budget: usize,
    buffers: HashMap<u64, MtlBuffer>,
}

impl MtlDevice {
    fn new(budget: usize) -> Self {
        Self { allocated: 0, budget, buffers: HashMap::new() }
    }

    fn alloc(
        &mut self,
        size: usize,
        mode: StorageMode,
        alignment: usize,
    ) -> Result<u64, AllocError> {
        if size == 0 {
            return Err(AllocError::ZeroSize);
        }
        let aligned = align_up(size, alignment);
        if self.allocated + aligned > self.budget {
            return Err(AllocError::OutOfMemory);
        }
        let buf = MtlBuffer::new(size, mode, alignment);
        let id = buf.id;
        self.allocated += aligned;
        self.buffers.insert(id, buf);
        Ok(id)
    }

    fn dealloc(&mut self, id: u64) -> Result<(), AllocError> {
        let buf = self.buffers.remove(&id).ok_or(AllocError::InvalidBuffer)?;
        self.allocated -= buf.aligned_size;
        Ok(())
    }

    fn get(&self, id: u64) -> Option<&MtlBuffer> {
        self.buffers.get(&id)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut MtlBuffer> {
        self.buffers.get_mut(&id)
    }

    fn available(&self) -> usize {
        self.budget.saturating_sub(self.allocated)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AllocError {
    ZeroSize,
    OutOfMemory,
    InvalidBuffer,
    AlignmentViolation,
}

fn align_up(size: usize, alignment: usize) -> usize {
    if size == 0 {
        return 0;
    }
    debug_assert!(alignment.is_power_of_two());
    (size + alignment - 1) & !(alignment - 1)
}

// ── Buffer Pool ─────────────────────────────────────────────────────

/// Size-class binning: rounds up to next power-of-two bucket.
fn size_class(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    size.next_power_of_two()
}

#[derive(Debug)]
struct BufferPool {
    free: BTreeMap<usize, VecDeque<MtlBuffer>>,
    in_use: HashMap<u64, MtlBuffer>,
    max_cached: usize,
    cached_count: usize,
}

impl BufferPool {
    fn new(max_cached: usize) -> Self {
        Self { free: BTreeMap::new(), in_use: HashMap::new(), max_cached, cached_count: 0 }
    }

    fn acquire(&mut self, size: usize, mode: StorageMode) -> MtlBuffer {
        let cls = size_class(size);
        if let Some(queue) = self.free.get_mut(&cls) {
            if let Some(buf) = queue.pop_front() {
                self.cached_count -= 1;
                self.in_use.insert(buf.id, buf.clone());
                return buf;
            }
        }
        let buf = MtlBuffer::new(size, mode, ALIGN_256);
        self.in_use.insert(buf.id, buf.clone());
        buf
    }

    fn release(&mut self, id: u64) -> bool {
        if let Some(buf) = self.in_use.remove(&id) {
            if self.cached_count < self.max_cached {
                let cls = size_class(buf.size);
                self.free.entry(cls).or_default().push_back(buf);
                self.cached_count += 1;
            }
            // else: buffer is dropped (eviction)
            true
        } else {
            false
        }
    }

    fn cached_count(&self) -> usize {
        self.cached_count
    }

    fn evict_all(&mut self) {
        self.free.clear();
        self.cached_count = 0;
    }
}

// ── Transfer engine ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransferDir {
    HostToDevice,
    DeviceToHost,
}

#[derive(Debug, Clone)]
struct TransferOp {
    src_id: u64,
    dst_id: u64,
    dir: TransferDir,
    size: usize,
    fence_value: u64,
}

#[derive(Debug)]
struct TransferEngine {
    ops: Vec<TransferOp>,
    next_fence: u64,
}

impl TransferEngine {
    fn new() -> Self {
        Self { ops: Vec::new(), next_fence: 1 }
    }

    fn enqueue(&mut self, src: u64, dst: u64, dir: TransferDir, size: usize) -> u64 {
        let fence = self.next_fence;
        self.next_fence += 1;
        self.ops.push(TransferOp { src_id: src, dst_id: dst, dir, size, fence_value: fence });
        fence
    }

    fn completed_fences(&self) -> Vec<u64> {
        self.ops.iter().map(|op| op.fence_value).collect()
    }
}

// ── Ring buffer ─────────────────────────────────────────────────────

#[derive(Debug)]
struct RingBuffer {
    data: Vec<u8>,
    capacity: usize,
    write_pos: usize,
    frame_offsets: VecDeque<usize>,
    max_frames: usize,
}

impl RingBuffer {
    fn new(capacity: usize, max_frames: usize) -> Self {
        Self {
            data: vec![0u8; capacity],
            capacity,
            write_pos: 0,
            frame_offsets: VecDeque::new(),
            max_frames,
        }
    }

    fn push_frame(&mut self, payload: &[u8]) -> Option<usize> {
        if payload.len() > self.capacity {
            return None;
        }
        // evict oldest if at limit
        if self.frame_offsets.len() >= self.max_frames {
            self.frame_offsets.pop_front();
        }
        let offset = self.write_pos;
        let end = offset + payload.len();
        if end <= self.capacity {
            self.data[offset..end].copy_from_slice(payload);
        } else {
            // wrap around
            let first = self.capacity - offset;
            self.data[offset..].copy_from_slice(&payload[..first]);
            self.data[..payload.len() - first].copy_from_slice(&payload[first..]);
        }
        self.write_pos = end % self.capacity;
        self.frame_offsets.push_back(offset);
        Some(offset)
    }

    fn frame_count(&self) -> usize {
        self.frame_offsets.len()
    }
}

// ── Alias tracker ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Lifetime {
    start: u64,
    end: u64,
}

impl Lifetime {
    fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[derive(Debug)]
struct AliasTracker {
    entries: Vec<(u64, Lifetime)>,
}

impl AliasTracker {
    fn new() -> Self {
        Self { entries: Vec::new() }
    }

    fn register(&mut self, buffer_id: u64, lt: Lifetime) -> Result<(), &'static str> {
        let conflict = self.entries.iter().any(|(_, existing)| existing.overlaps(&lt));
        if conflict {
            return Err("lifetime overlap detected");
        }
        self.entries.push((buffer_id, lt));
        Ok(())
    }

    fn can_alias(&self, lt: &Lifetime) -> bool {
        !self.entries.iter().any(|(_, e)| e.overlaps(lt))
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ── Argument buffer ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgTier {
    Tier1,
    Tier2,
}

#[derive(Debug, Clone)]
struct ArgEntry {
    index: u32,
    offset: usize,
    size: usize,
}

#[derive(Debug)]
struct ArgumentBuffer {
    tier: ArgTier,
    entries: Vec<ArgEntry>,
    data: Vec<u8>,
    alignment: usize,
}

impl ArgumentBuffer {
    fn new(tier: ArgTier) -> Self {
        let alignment = match tier {
            ArgTier::Tier1 => ALIGN_16,
            ArgTier::Tier2 => ALIGN_256,
        };
        Self { tier, entries: Vec::new(), data: Vec::new(), alignment }
    }

    fn encode_buffer(&mut self, index: u32, size: usize) {
        let offset = align_up(self.data.len(), self.alignment);
        self.data.resize(offset + size, 0);
        self.entries.push(ArgEntry { index, offset, size });
    }

    fn entry_count(&self) -> usize {
        self.entries.len()
    }

    fn total_size(&self) -> usize {
        self.data.len()
    }

    fn entry_at(&self, index: u32) -> Option<&ArgEntry> {
        self.entries.iter().find(|e| e.index == index)
    }
}

#[derive(Debug)]
struct IndirectCommandBuffer {
    commands: Vec<IndirectCommand>,
    max_commands: usize,
}

#[derive(Debug, Clone)]
struct IndirectCommand {
    kernel_index: u32,
    buffer_bindings: Vec<(u32, u64)>, // (index, buffer_id)
    threadgroups: [u32; 3],
}

impl IndirectCommandBuffer {
    fn new(max_commands: usize) -> Self {
        Self { commands: Vec::new(), max_commands }
    }

    fn push(&mut self, cmd: IndirectCommand) -> bool {
        if self.commands.len() >= self.max_commands {
            return false;
        }
        self.commands.push(cmd);
        true
    }

    fn len(&self) -> usize {
        self.commands.len()
    }
}

// ═══════════════════════════════════════════════════════════════════
// 1. Buffer Allocation Tests
// ═══════════════════════════════════════════════════════════════════

mod allocation {
    use super::*;

    #[test]
    fn alloc_basic_shared() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(1024, StorageMode::Shared, ALIGN_256).unwrap();
        let buf = dev.get(id).unwrap();
        assert_eq!(buf.size, 1024);
        assert_eq!(buf.storage_mode, StorageMode::Shared);
    }

    #[test]
    fn alloc_basic_private() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(2048, StorageMode::Private, ALIGN_256).unwrap();
        assert_eq!(dev.get(id).unwrap().storage_mode, StorageMode::Private);
    }

    #[test]
    fn alloc_basic_managed() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(4096, StorageMode::Managed, ALIGN_256).unwrap();
        assert_eq!(dev.get(id).unwrap().storage_mode, StorageMode::Managed);
    }

    #[test]
    fn alloc_16_byte_alignment() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(100, StorageMode::Shared, ALIGN_16).unwrap();
        let buf = dev.get(id).unwrap();
        assert!(buf.aligned_size.is_multiple_of(ALIGN_16));
        assert!(buf.aligned_size >= 100);
    }

    #[test]
    fn alloc_256_byte_alignment() {
        let sizes: Vec<usize> = vec![1, 127, 255, 256, 257, 511, 512];
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        for &sz in &sizes {
            let id = dev.alloc(sz, StorageMode::Shared, ALIGN_256).unwrap();
            let buf = dev.get(id).unwrap();
            assert!(
                buf.aligned_size.is_multiple_of(ALIGN_256),
                "size {sz} → aligned {} not multiple of 256",
                buf.aligned_size
            );
        }
    }

    #[test]
    fn alloc_4096_byte_alignment() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(5000, StorageMode::Shared, ALIGN_4096).unwrap();
        let buf = dev.get(id).unwrap();
        assert!(buf.aligned_size.is_multiple_of(ALIGN_4096));
        assert_eq!(buf.aligned_size, 8192);
    }

    #[test]
    fn alloc_zero_size_rejected() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let err = dev.alloc(0, StorageMode::Shared, ALIGN_256).unwrap_err();
        assert_eq!(err, AllocError::ZeroSize);
    }

    #[test]
    fn alloc_exact_budget() {
        let budget = 4096;
        let mut dev = MtlDevice::new(budget);
        let id = dev.alloc(budget, StorageMode::Shared, ALIGN_256).unwrap();
        assert!(dev.get(id).is_some());
        assert_eq!(dev.available(), 0);
    }

    #[test]
    fn alloc_over_budget_fails() {
        let mut dev = MtlDevice::new(1024);
        let err = dev.alloc(2048, StorageMode::Shared, ALIGN_256).unwrap_err();
        assert_eq!(err, AllocError::OutOfMemory);
    }

    #[test]
    fn alloc_cumulative_budget_tracking() {
        let mut dev = MtlDevice::new(1024);
        dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        // budget exhausted
        let err = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap_err();
        assert_eq!(err, AllocError::OutOfMemory);
    }

    #[test]
    fn dealloc_frees_budget() {
        let mut dev = MtlDevice::new(512);
        let id = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        assert_eq!(dev.available(), 256);
        dev.dealloc(id).unwrap();
        assert_eq!(dev.available(), 512);
    }

    #[test]
    fn dealloc_invalid_id_errors() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let err = dev.dealloc(9999).unwrap_err();
        assert_eq!(err, AllocError::InvalidBuffer);
    }

    #[test]
    fn alloc_unique_ids() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let ids: Vec<u64> =
            (0..20).map(|_| dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len());
    }

    #[test]
    fn alloc_data_initialised_to_zero() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(512, StorageMode::Shared, ALIGN_256).unwrap();
        let buf = dev.get(id).unwrap();
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn alloc_aligned_size_ge_requested() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let test_sizes: Vec<usize> =
            (1..=17).chain([100, 255, 513, 1023, 4097].iter().copied()).collect();
        for sz in test_sizes {
            let id = dev.alloc(sz, StorageMode::Shared, ALIGN_256).unwrap();
            let buf = dev.get(id).unwrap();
            assert!(buf.aligned_size >= sz, "aligned {} < requested {sz}", buf.aligned_size);
        }
    }

    #[test]
    fn alloc_page_aligned_large_buffer() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(PAGE_SIZE + 1, StorageMode::Shared, PAGE_SIZE).unwrap();
        let buf = dev.get(id).unwrap();
        assert!(buf.aligned_size.is_multiple_of(PAGE_SIZE));
        assert_eq!(buf.aligned_size, PAGE_SIZE * 2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Buffer Pool Tests
// ═══════════════════════════════════════════════════════════════════

mod pool {
    use super::*;

    #[test]
    fn pool_create_empty() {
        let pool = BufferPool::new(16);
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn pool_acquire_creates_buffer() {
        let mut pool = BufferPool::new(16);
        let buf = pool.acquire(1024, StorageMode::Shared);
        assert_eq!(buf.size, 1024);
    }

    #[test]
    fn pool_release_caches_buffer() {
        let mut pool = BufferPool::new(16);
        let buf = pool.acquire(1024, StorageMode::Shared);
        let id = buf.id;
        pool.release(id);
        assert_eq!(pool.cached_count(), 1);
    }

    #[test]
    fn pool_reuse_after_release() {
        let mut pool = BufferPool::new(16);
        let buf = pool.acquire(1024, StorageMode::Shared);
        let first_id = buf.id;
        pool.release(first_id);

        let reused = pool.acquire(1024, StorageMode::Shared);
        assert_eq!(reused.id, first_id);
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn pool_size_class_binning() {
        assert_eq!(size_class(1), 1);
        assert_eq!(size_class(2), 2);
        assert_eq!(size_class(3), 4);
        assert_eq!(size_class(5), 8);
        assert_eq!(size_class(100), 128);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 2048);
    }

    #[test]
    fn pool_same_class_reuse() {
        let mut pool = BufferPool::new(16);
        // 900 and 1000 both round up to class 1024
        let buf_a = pool.acquire(900, StorageMode::Shared);
        let id_a = buf_a.id;
        pool.release(id_a);

        let buf_b = pool.acquire(1000, StorageMode::Shared);
        assert_eq!(buf_b.id, id_a, "same size class should reuse");
    }

    #[test]
    fn pool_different_class_no_reuse() {
        let mut pool = BufferPool::new(16);
        let buf_a = pool.acquire(100, StorageMode::Shared);
        let id_a = buf_a.id;
        pool.release(id_a);

        // 2000 → class 2048, won't match class 128
        let buf_b = pool.acquire(2000, StorageMode::Shared);
        assert_ne!(buf_b.id, id_a);
    }

    #[test]
    fn pool_eviction_under_pressure() {
        let mut pool = BufferPool::new(2);
        let ids: Vec<u64> = (0..3)
            .map(|_| {
                let b = pool.acquire(256, StorageMode::Shared);
                b.id
            })
            .collect();
        // Release all three, but max_cached=2
        for &id in &ids {
            pool.release(id);
        }
        assert_eq!(pool.cached_count(), 2);
    }

    #[test]
    fn pool_evict_all() {
        let mut pool = BufferPool::new(16);
        for _ in 0..10 {
            let b = pool.acquire(512, StorageMode::Shared);
            pool.release(b.id);
        }
        assert!(pool.cached_count() > 0);
        pool.evict_all();
        assert_eq!(pool.cached_count(), 0);
    }

    #[test]
    fn pool_release_unknown_id() {
        let mut pool = BufferPool::new(16);
        assert!(!pool.release(99999));
    }

    #[test]
    fn pool_multiple_size_classes_independent() {
        let mut pool = BufferPool::new(100);
        let small = pool.acquire(64, StorageMode::Shared);
        let large = pool.acquire(4096, StorageMode::Shared);
        pool.release(small.id);
        pool.release(large.id);
        assert_eq!(pool.cached_count(), 2);

        // Acquire small → reuses small class
        let s2 = pool.acquire(60, StorageMode::Shared);
        assert_eq!(s2.id, small.id);
        // Large class still cached
        assert_eq!(pool.cached_count(), 1);
    }

    #[test]
    fn pool_fifo_reuse_order() {
        let mut pool = BufferPool::new(16);
        let a = pool.acquire(256, StorageMode::Shared);
        let b = pool.acquire(256, StorageMode::Shared);
        let id_a = a.id;
        let id_b = b.id;
        pool.release(id_a);
        pool.release(id_b);

        // FIFO: a was released first → reused first
        let c = pool.acquire(256, StorageMode::Shared);
        assert_eq!(c.id, id_a);
    }

    #[test]
    fn pool_thread_safety_simulation() {
        let pool = Arc::new(Mutex::new(BufferPool::new(64)));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    let mut p = pool.lock().unwrap();
                    let buf = p.acquire(512, StorageMode::Shared);
                    let id = buf.id;
                    p.release(id);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let p = pool.lock().unwrap();
        assert!(p.cached_count() <= 64);
    }

    #[test]
    fn pool_stress_acquire_release_cycle() {
        let mut pool = BufferPool::new(32);
        let mut ids = Vec::new();
        for _ in 0..100 {
            let buf = pool.acquire(128, StorageMode::Shared);
            ids.push(buf.id);
        }
        for id in ids {
            pool.release(id);
        }
        // max_cached = 32
        assert_eq!(pool.cached_count(), 32);
    }

    #[test]
    fn pool_zero_capacity_never_caches() {
        let mut pool = BufferPool::new(0);
        let buf = pool.acquire(256, StorageMode::Shared);
        pool.release(buf.id);
        assert_eq!(pool.cached_count(), 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Host-Device Transfer Tests
// ═══════════════════════════════════════════════════════════════════

mod transfer {
    use super::*;

    #[test]
    fn upload_staging_shared_buffer() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let staging = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        let payload = vec![0xABu8; 128];
        dev.get_mut(staging).unwrap().write(0, &payload);
        let data = dev.get(staging).unwrap().read(0, 128);
        assert!(data.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn download_readback_buffer() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let gpu_buf = dev.alloc(512, StorageMode::Private, ALIGN_256).unwrap();
        let readback = dev.alloc(512, StorageMode::Shared, ALIGN_256).unwrap();

        // Simulate GPU writing result into private buffer
        dev.get_mut(gpu_buf).unwrap().write(0, &[0xCD; 256]);
        // Simulate blit copy to readback
        let src_data = dev.get(gpu_buf).unwrap().read(0, 256).to_vec();
        dev.get_mut(readback).unwrap().write(0, &src_data);

        let result = dev.get(readback).unwrap().read(0, 256);
        assert!(result.iter().all(|&b| b == 0xCD));
    }

    #[test]
    fn transfer_fence_ordering() {
        let mut engine = TransferEngine::new();
        let f1 = engine.enqueue(1, 2, TransferDir::HostToDevice, 1024);
        let f2 = engine.enqueue(2, 3, TransferDir::DeviceToHost, 512);
        let f3 = engine.enqueue(4, 5, TransferDir::HostToDevice, 2048);

        assert!(f1 < f2);
        assert!(f2 < f3);
    }

    #[test]
    fn transfer_fence_monotonic() {
        let mut engine = TransferEngine::new();
        let fences: Vec<u64> =
            (0..20).map(|i| engine.enqueue(i, i + 100, TransferDir::HostToDevice, 256)).collect();
        assert!(fences.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn transfer_completed_fences_returns_all() {
        let mut engine = TransferEngine::new();
        for i in 0..5 {
            engine.enqueue(i, i + 10, TransferDir::HostToDevice, 64);
        }
        assert_eq!(engine.completed_fences().len(), 5);
    }

    #[test]
    fn double_buffering_pattern() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let buf_a = dev.alloc(1024, StorageMode::Shared, ALIGN_256).unwrap();
        let buf_b = dev.alloc(1024, StorageMode::Shared, ALIGN_256).unwrap();

        // Frame 0: write A, GPU reads A
        dev.get_mut(buf_a).unwrap().write(0, &[1u8; 512]);
        // Frame 1: write B, GPU reads B; CPU can read A's result
        dev.get_mut(buf_b).unwrap().write(0, &[2u8; 512]);

        let a_data = dev.get(buf_a).unwrap().read(0, 1);
        let b_data = dev.get(buf_b).unwrap().read(0, 1);
        assert_eq!(a_data[0], 1);
        assert_eq!(b_data[0], 2);
    }

    #[test]
    fn triple_buffering_pattern() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let bufs: Vec<u64> =
            (0..3).map(|_| dev.alloc(512, StorageMode::Shared, ALIGN_256).unwrap()).collect();
        for (frame, &buf_id) in bufs.iter().enumerate() {
            dev.get_mut(buf_id).unwrap().write(0, &[frame as u8; 256]);
        }
        for (frame, &buf_id) in bufs.iter().enumerate() {
            assert_eq!(dev.get(buf_id).unwrap().read(0, 1)[0], frame as u8);
        }
    }

    #[test]
    fn ring_buffer_basic_push() {
        let mut ring = RingBuffer::new(1024, 4);
        let offset = ring.push_frame(&[0xAA; 100]);
        assert!(offset.is_some());
        assert_eq!(ring.frame_count(), 1);
    }

    #[test]
    fn ring_buffer_rotation() {
        let mut ring = RingBuffer::new(256, 8);
        for i in 0u8..8 {
            let payload = vec![i; 30];
            ring.push_frame(&payload).unwrap();
        }
        assert_eq!(ring.frame_count(), 8);
    }

    #[test]
    fn ring_buffer_evicts_oldest_frame() {
        let mut ring = RingBuffer::new(1024, 3);
        ring.push_frame(&[1; 64]).unwrap();
        ring.push_frame(&[2; 64]).unwrap();
        ring.push_frame(&[3; 64]).unwrap();
        // 4th push evicts frame 0
        ring.push_frame(&[4; 64]).unwrap();
        assert_eq!(ring.frame_count(), 3);
    }

    #[test]
    fn ring_buffer_oversized_payload_rejected() {
        let mut ring = RingBuffer::new(64, 4);
        assert!(ring.push_frame(&[0; 128]).is_none());
        assert_eq!(ring.frame_count(), 0);
    }

    #[test]
    fn ring_buffer_wrap_around() {
        let mut ring = RingBuffer::new(100, 10);
        // Push several frames that wrap around
        for i in 0u8..5 {
            ring.push_frame(&vec![i; 30]).unwrap();
        }
        assert_eq!(ring.frame_count(), 5);
    }

    #[test]
    fn transfer_direction_variants() {
        let mut engine = TransferEngine::new();
        let f_up = engine.enqueue(1, 2, TransferDir::HostToDevice, 256);
        let f_down = engine.enqueue(3, 4, TransferDir::DeviceToHost, 256);
        let ops = &engine.ops;
        assert_eq!(ops[0].dir, TransferDir::HostToDevice);
        assert_eq!(ops[1].dir, TransferDir::DeviceToHost);
        assert!(f_up < f_down);
    }

    #[test]
    fn transfer_size_recorded() {
        let mut engine = TransferEngine::new();
        engine.enqueue(1, 2, TransferDir::HostToDevice, 4096);
        assert_eq!(engine.ops[0].size, 4096);
    }

    #[test]
    fn staging_buffer_write_then_read_roundtrip() {
        let mut dev = MtlDevice::new(SIMULATED_BUDGET);
        let id = dev.alloc(1024, StorageMode::Shared, ALIGN_256).unwrap();
        let pattern: Vec<u8> = (0..128).map(|i| (i & 0xFF) as u8).collect();
        dev.get_mut(id).unwrap().write(0, &pattern);
        let readback = dev.get(id).unwrap().read(0, 128);
        assert_eq!(readback, &pattern[..]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Memory Pressure Tests
// ═══════════════════════════════════════════════════════════════════

mod pressure {
    use super::*;

    #[test]
    fn alloc_within_budget_succeeds() {
        let budget = 64 * 1024;
        let mut dev = MtlDevice::new(budget);
        let id = dev.alloc(32 * 1024, StorageMode::Shared, ALIGN_256).unwrap();
        assert!(dev.get(id).is_some());
        assert!(dev.available() > 0);
    }

    #[test]
    fn alloc_exactly_at_budget_succeeds() {
        let mut dev = MtlDevice::new(ALIGN_256);
        let id = dev.alloc(ALIGN_256, StorageMode::Shared, ALIGN_256).unwrap();
        assert_eq!(dev.available(), 0);
        assert!(dev.get(id).is_some());
    }

    #[test]
    fn alloc_over_budget_returns_oom() {
        let mut dev = MtlDevice::new(ALIGN_256);
        dev.alloc(ALIGN_256, StorageMode::Shared, ALIGN_256).unwrap();
        let err = dev.alloc(1, StorageMode::Shared, ALIGN_256).unwrap_err();
        assert_eq!(err, AllocError::OutOfMemory);
    }

    #[test]
    fn oom_recovery_after_dealloc() {
        let mut dev = MtlDevice::new(512);
        let id_a = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        let id_b = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        assert_eq!(
            dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap_err(),
            AllocError::OutOfMemory
        );
        dev.dealloc(id_a).unwrap();
        // Now there's room
        let id_c = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        assert!(dev.get(id_c).is_some());
        dev.dealloc(id_b).unwrap();
    }

    #[test]
    fn graceful_degradation_smaller_alloc() {
        let mut dev = MtlDevice::new(1024);
        dev.alloc(768, StorageMode::Shared, ALIGN_256).unwrap();
        // Can't fit 512, but can fit 256
        assert_eq!(
            dev.alloc(512, StorageMode::Shared, ALIGN_256).unwrap_err(),
            AllocError::OutOfMemory
        );
        let id = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        assert!(dev.get(id).is_some());
    }

    #[test]
    fn purgeable_buffer_volatile() {
        let mut buf = MtlBuffer::new(1024, StorageMode::Shared, ALIGN_256);
        assert_eq!(buf.purge_state, PurgeState::NonVolatile);
        buf.purge_state = PurgeState::Volatile;
        assert_eq!(buf.purge_state, PurgeState::Volatile);
    }

    #[test]
    fn purgeable_buffer_purged_state() {
        let mut buf = MtlBuffer::new(1024, StorageMode::Shared, ALIGN_256);
        buf.purge_state = PurgeState::Volatile;
        // Simulate system purge
        buf.purge_state = PurgeState::Purged;
        assert_eq!(buf.purge_state, PurgeState::Purged);
    }

    #[test]
    fn purgeable_recovery_re_alloc() {
        let mut buf = MtlBuffer::new(512, StorageMode::Shared, ALIGN_256);
        buf.purge_state = PurgeState::Purged;
        // Recovery: re-populate data
        buf.data = vec![0u8; buf.aligned_size];
        buf.purge_state = PurgeState::NonVolatile;
        assert_eq!(buf.purge_state, PurgeState::NonVolatile);
        assert_eq!(buf.data.len(), buf.aligned_size);
    }

    #[test]
    fn budget_tracking_across_modes() {
        let mut dev = MtlDevice::new(2048);
        dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        dev.alloc(256, StorageMode::Private, ALIGN_256).unwrap();
        dev.alloc(256, StorageMode::Managed, ALIGN_256).unwrap();
        assert_eq!(dev.available(), 2048 - 768);
    }

    #[test]
    fn repeated_alloc_dealloc_stable() {
        let mut dev = MtlDevice::new(1024);
        for _ in 0..50 {
            let id = dev.alloc(512, StorageMode::Shared, ALIGN_256).unwrap();
            dev.dealloc(id).unwrap();
        }
        assert_eq!(dev.available(), 1024);
    }

    #[test]
    fn watermark_tracking() {
        let mut dev = MtlDevice::new(4096);
        let mut peak = 0usize;
        let mut ids = Vec::new();
        for _ in 0..8 {
            let id = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
            ids.push(id);
            let used = dev.budget - dev.available();
            peak = peak.max(used);
        }
        assert_eq!(peak, 2048);
        for id in ids {
            dev.dealloc(id).unwrap();
        }
        assert_eq!(dev.available(), 4096);
    }

    #[test]
    fn oom_does_not_leak_memory() {
        let mut dev = MtlDevice::new(512);
        let id = dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        let avail_before = dev.available();
        let _ = dev.alloc(512, StorageMode::Shared, ALIGN_256);
        // Failed alloc should not change available memory
        assert_eq!(dev.available(), avail_before);
        dev.dealloc(id).unwrap();
    }

    #[test]
    fn alloc_many_small_buffers() {
        let mut dev = MtlDevice::new(256 * 100);
        let ids: Vec<u64> =
            (0..100).map(|_| dev.alloc(128, StorageMode::Shared, ALIGN_256).unwrap()).collect();
        assert_eq!(ids.len(), 100);
        assert_eq!(dev.available(), 0);
    }

    #[test]
    fn available_never_negative() {
        let mut dev = MtlDevice::new(256);
        dev.alloc(256, StorageMode::Shared, ALIGN_256).unwrap();
        assert_eq!(dev.available(), 0);
        // Trying to alloc more fails but available stays 0
        let _ = dev.alloc(1, StorageMode::Shared, ALIGN_256);
        assert_eq!(dev.available(), 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Buffer Aliasing Tests
// ═══════════════════════════════════════════════════════════════════

mod aliasing {
    use super::*;

    #[test]
    fn non_overlapping_lifetimes_succeed() {
        let mut tracker = AliasTracker::new();
        tracker.register(1, Lifetime { start: 0, end: 5 }).unwrap();
        tracker.register(1, Lifetime { start: 5, end: 10 }).unwrap();
        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn overlapping_lifetimes_rejected() {
        let mut tracker = AliasTracker::new();
        tracker.register(1, Lifetime { start: 0, end: 10 }).unwrap();
        let err = tracker.register(2, Lifetime { start: 5, end: 15 }).unwrap_err();
        assert_eq!(err, "lifetime overlap detected");
    }

    #[test]
    fn adjacent_lifetimes_no_overlap() {
        let lt_a = Lifetime { start: 0, end: 5 };
        let lt_b = Lifetime { start: 5, end: 10 };
        assert!(!lt_a.overlaps(&lt_b));
        assert!(!lt_b.overlaps(&lt_a));
    }

    #[test]
    fn identical_lifetimes_overlap() {
        let lt = Lifetime { start: 3, end: 7 };
        assert!(lt.overlaps(&lt));
    }

    #[test]
    fn contained_lifetime_overlap() {
        let outer = Lifetime { start: 0, end: 100 };
        let inner = Lifetime { start: 10, end: 20 };
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    #[test]
    fn zero_length_lifetime_no_overlap() {
        let zero = Lifetime { start: 5, end: 5 };
        let other = Lifetime { start: 5, end: 10 };
        assert!(!zero.overlaps(&other));
    }

    #[test]
    fn can_alias_query() {
        let mut tracker = AliasTracker::new();
        tracker.register(1, Lifetime { start: 0, end: 5 }).unwrap();
        assert!(tracker.can_alias(&Lifetime { start: 5, end: 10 }));
        assert!(!tracker.can_alias(&Lifetime { start: 4, end: 6 }));
    }

    #[test]
    fn sequential_alias_chain() {
        let mut tracker = AliasTracker::new();
        for i in 0u64..10 {
            tracker.register(i, Lifetime { start: i * 10, end: (i + 1) * 10 }).unwrap();
        }
        assert_eq!(tracker.len(), 10);
    }

    #[test]
    fn alias_validates_gap_between_uses() {
        let mut tracker = AliasTracker::new();
        tracker.register(1, Lifetime { start: 0, end: 3 }).unwrap();
        // gap at [3..7]
        tracker.register(1, Lifetime { start: 7, end: 10 }).unwrap();
        // [3..7] should be available
        assert!(tracker.can_alias(&Lifetime { start: 3, end: 7 }));
    }

    #[test]
    fn hazard_detection_on_write_after_write() {
        let mut tracker = AliasTracker::new();
        tracker.register(1, Lifetime { start: 0, end: 10 }).unwrap();
        // Another write during the same lifetime window
        let result = tracker.register(2, Lifetime { start: 5, end: 8 });
        assert!(result.is_err());
    }

    #[test]
    fn hazard_detection_read_after_write() {
        let mut tracker = AliasTracker::new();
        // Write phase
        tracker.register(1, Lifetime { start: 0, end: 5 }).unwrap();
        // Read in overlapping window
        let result = tracker.register(2, Lifetime { start: 3, end: 7 });
        assert!(result.is_err());
    }

    #[test]
    fn temporal_reuse_same_buffer_id() {
        let mut tracker = AliasTracker::new();
        // Same buffer reused at different times
        tracker.register(42, Lifetime { start: 0, end: 100 }).unwrap();
        tracker.register(42, Lifetime { start: 200, end: 300 }).unwrap();
        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn empty_tracker_allows_any_lifetime() {
        let tracker = AliasTracker::new();
        assert!(tracker.can_alias(&Lifetime { start: 0, end: 1000 }));
    }

    #[test]
    fn many_non_overlapping_segments() {
        let mut tracker = AliasTracker::new();
        for i in 0u64..100 {
            tracker.register(i, Lifetime { start: i * 2, end: i * 2 + 1 }).unwrap();
        }
        assert_eq!(tracker.len(), 100);
        // Odd slots should be available
        assert!(tracker.can_alias(&Lifetime { start: 1, end: 2 }));
    }

    #[test]
    fn partial_overlap_at_start() {
        let lt_a = Lifetime { start: 5, end: 15 };
        let lt_b = Lifetime { start: 0, end: 6 };
        assert!(lt_a.overlaps(&lt_b));
    }

    #[test]
    fn partial_overlap_at_end() {
        let lt_a = Lifetime { start: 0, end: 10 };
        let lt_b = Lifetime { start: 9, end: 20 };
        assert!(lt_a.overlaps(&lt_b));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Argument Buffer Tests
// ═══════════════════════════════════════════════════════════════════

mod argument_buffer {
    use super::*;

    #[test]
    fn tier1_alignment_is_16() {
        let ab = ArgumentBuffer::new(ArgTier::Tier1);
        assert_eq!(ab.alignment, ALIGN_16);
    }

    #[test]
    fn tier2_alignment_is_256() {
        let ab = ArgumentBuffer::new(ArgTier::Tier2);
        assert_eq!(ab.alignment, ALIGN_256);
    }

    #[test]
    fn encode_single_buffer_tier1() {
        let mut ab = ArgumentBuffer::new(ArgTier::Tier1);
        ab.encode_buffer(0, 64);
        assert_eq!(ab.entry_count(), 1);
        let entry = ab.entry_at(0).unwrap();
        assert!(entry.offset.is_multiple_of(ALIGN_16));
        assert_eq!(entry.size, 64);
    }

    #[test]
    fn encode_single_buffer_tier2() {
        let mut ab = ArgumentBuffer::new(ArgTier::Tier2);
        ab.encode_buffer(0, 128);
        assert_eq!(ab.entry_count(), 1);
        let entry = ab.entry_at(0).unwrap();
        assert!(entry.offset.is_multiple_of(ALIGN_256));
    }

    #[test]
    fn encode_multiple_entries_aligned() {
        let mut ab = ArgumentBuffer::new(ArgTier::Tier2);
        ab.encode_buffer(0, 64);
        ab.encode_buffer(1, 128);
        ab.encode_buffer(2, 32);
        assert_eq!(ab.entry_count(), 3);
        for entry in &ab.entries {
            assert!(
                entry.offset.is_multiple_of(ALIGN_256),
                "entry {} offset {} not aligned to 256",
                entry.index,
                entry.offset
            );
        }
    }

    #[test]
    fn encode_entries_non_overlapping() {
        let mut ab = ArgumentBuffer::new(ArgTier::Tier1);
        ab.encode_buffer(0, 100);
        ab.encode_buffer(1, 200);
        ab.encode_buffer(2, 50);
        for pair in ab.entries.windows(2) {
            let end_prev = pair[0].offset + pair[0].size;
            assert!(
                pair[1].offset >= end_prev,
                "entry {} overlaps entry {}",
                pair[1].index,
                pair[0].index
            );
        }
    }

    #[test]
    fn entry_at_missing_index_returns_none() {
        let ab = ArgumentBuffer::new(ArgTier::Tier1);
        assert!(ab.entry_at(0).is_none());
    }

    #[test]
    fn total_size_grows_with_entries() {
        let mut ab = ArgumentBuffer::new(ArgTier::Tier1);
        let s0 = ab.total_size();
        ab.encode_buffer(0, 32);
        assert!(ab.total_size() > s0);
        let s1 = ab.total_size();
        ab.encode_buffer(1, 64);
        assert!(ab.total_size() > s1);
    }

    #[test]
    fn tier1_vs_tier2_size_difference() {
        let mut t1 = ArgumentBuffer::new(ArgTier::Tier1);
        let mut t2 = ArgumentBuffer::new(ArgTier::Tier2);
        // Encode same entries in both
        for i in 0..4 {
            t1.encode_buffer(i, 32);
            t2.encode_buffer(i, 32);
        }
        // Tier-2 should be larger due to 256-byte alignment padding
        assert!(
            t2.total_size() >= t1.total_size(),
            "tier2 {} < tier1 {}",
            t2.total_size(),
            t1.total_size()
        );
    }

    #[test]
    fn indirect_command_buffer_creation() {
        let icb = IndirectCommandBuffer::new(64);
        assert_eq!(icb.len(), 0);
    }

    #[test]
    fn indirect_command_buffer_push() {
        let mut icb = IndirectCommandBuffer::new(64);
        let cmd = IndirectCommand {
            kernel_index: 0,
            buffer_bindings: vec![(0, 100), (1, 200)],
            threadgroups: [8, 1, 1],
        };
        assert!(icb.push(cmd));
        assert_eq!(icb.len(), 1);
    }

    #[test]
    fn indirect_command_buffer_capacity_limit() {
        let mut icb = IndirectCommandBuffer::new(3);
        for i in 0..3 {
            assert!(icb.push(IndirectCommand {
                kernel_index: i,
                buffer_bindings: vec![],
                threadgroups: [1, 1, 1],
            }));
        }
        assert!(!icb.push(IndirectCommand {
            kernel_index: 99,
            buffer_bindings: vec![],
            threadgroups: [1, 1, 1],
        }));
        assert_eq!(icb.len(), 3);
    }

    #[test]
    fn indirect_command_preserves_bindings() {
        let mut icb = IndirectCommandBuffer::new(8);
        let bindings = vec![(0, 10), (1, 20), (2, 30)];
        icb.push(IndirectCommand {
            kernel_index: 5,
            buffer_bindings: bindings.clone(),
            threadgroups: [4, 2, 1],
        });
        let cmd = &icb.commands[0];
        assert_eq!(cmd.kernel_index, 5);
        assert_eq!(cmd.buffer_bindings, bindings);
        assert_eq!(cmd.threadgroups, [4, 2, 1]);
    }

    #[test]
    fn argument_buffer_empty_initial_state() {
        let ab = ArgumentBuffer::new(ArgTier::Tier1);
        assert_eq!(ab.entry_count(), 0);
        assert_eq!(ab.total_size(), 0);
    }

    #[test]
    fn encode_large_entry_count() {
        let mut ab = ArgumentBuffer::new(ArgTier::Tier1);
        for i in 0..32 {
            ab.encode_buffer(i, 16);
        }
        assert_eq!(ab.entry_count(), 32);
        // All offsets aligned
        assert!(ab.entries.iter().all(|e| e.offset.is_multiple_of(ALIGN_16)));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Alignment helper unit tests
// ═══════════════════════════════════════════════════════════════════

mod alignment_helpers {
    use super::*;

    #[test]
    fn align_up_zero() {
        assert_eq!(align_up(0, 256), 0);
    }

    #[test]
    fn align_up_exact_multiple() {
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(512, 256), 512);
    }

    #[test]
    fn align_up_rounds_up() {
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn align_up_various_alignments() {
        assert_eq!(align_up(17, ALIGN_16), 32);
        assert_eq!(align_up(4000, ALIGN_4096), 4096);
        assert_eq!(align_up(4097, ALIGN_4096), 8192);
    }

    #[test]
    fn size_class_is_power_of_two() {
        for sz in [1, 2, 3, 7, 8, 9, 100, 1000, 1024, 1025] {
            let cls = size_class(sz);
            assert!(cls.is_power_of_two(), "size_class({sz}) = {cls} is not power of two");
        }
    }

    #[test]
    fn size_class_ge_input() {
        for sz in 1..=2048 {
            assert!(size_class(sz) >= sz, "size_class({sz}) = {} < {sz}", size_class(sz));
        }
    }

    #[test]
    fn size_class_zero() {
        assert_eq!(size_class(0), 0);
    }
}
