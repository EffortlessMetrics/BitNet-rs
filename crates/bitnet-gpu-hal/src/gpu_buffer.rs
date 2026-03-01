//! Module stub - implementation pending merge from feature branch
//! GPU buffer management with pooling, pinned memory, and async transfers.
//!
//! Provides [`GpuBuffer`] (device-side allocation), [`BufferPool`] (reuse via
//! free-list), [`PinnedBuffer`] / [`StagingBuffer`] (host-pinned DMA), and
//! [`TransferManager`] for host↔device copy scheduling.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ── ID generation ───────────────────────────────────────────────────────────

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

// ── BufferConfig ────────────────────────────────────────────────────────────

/// Settings for GPU buffer allocation.
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Required alignment in bytes (must be a power of two).
    pub alignment: usize,
    /// Whether to use host-pinned memory for staging buffers.
    pub use_pinned: bool,
    /// Whether the buffer pool is enabled.
    pub enable_pooling: bool,
    /// Maximum pool capacity in bytes.
    pub pool_size_bytes: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            alignment: 256,
            use_pinned: false,
            enable_pooling: true,
            pool_size_bytes: 256 * 1024 * 1024, // 256 MiB
        }
    }
}

impl BufferConfig {
    /// Round `size` up to the configured alignment.
    #[must_use]
    pub const fn align_up(&self, size: usize) -> usize {
        let mask = self.alignment - 1;
        (size + mask) & !mask
    }
}

// ── BufferUsage ─────────────────────────────────────────────────────────────

/// Intended usage pattern for a GPU buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// Device-side read-only (e.g. weights).
    ReadOnly,
    /// Device-side write-only (e.g. output activations).
    WriteOnly,
    /// General-purpose read/write.
    ReadWrite,
    /// Kernel scratch / intermediate.
    Kernel,
    /// Used for host↔device DMA transfers.
    Transfer,
    /// Staging buffer on the host side.
    Staging,
}

impl fmt::Display for BufferUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "ReadOnly"),
            Self::WriteOnly => write!(f, "WriteOnly"),
            Self::ReadWrite => write!(f, "ReadWrite"),
            Self::Kernel => write!(f, "Kernel"),
            Self::Transfer => write!(f, "Transfer"),
            Self::Staging => write!(f, "Staging"),
        }
    }
}

// ── GpuBuffer ───────────────────────────────────────────────────────────────

/// A GPU-side buffer allocation.
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Unique identifier.
    pub id: u64,
    /// Size in bytes (aligned).
    pub size: usize,
    /// Alignment in bytes.
    pub alignment: usize,
    /// Usage hint.
    pub usage: BufferUsage,
    /// Whether the buffer is currently host-mapped.
    pub is_mapped: bool,
    /// Logical offset within a larger device allocation (for sub-allocation).
    pub device_ptr_offset: u64,
}

impl GpuBuffer {
    /// Create a new buffer with the given parameters.
    #[must_use]
    pub fn new(size: usize, alignment: usize, usage: BufferUsage) -> Self {
        let mask = alignment - 1;
        let aligned = (size + mask) & !mask;
        Self {
            id: next_id(),
            size: aligned,
            alignment,
            usage,
            is_mapped: false,
            device_ptr_offset: 0,
        }
    }

    /// Create a buffer using a [`BufferConfig`].
    #[must_use]
    pub fn with_config(size: usize, usage: BufferUsage, config: &BufferConfig) -> Self {
        Self::new(size, config.alignment, usage)
    }

    /// Map the buffer for host access.
    pub const fn map(&mut self) {
        self.is_mapped = true;
    }

    /// Unmap the buffer.
    pub const fn unmap(&mut self) {
        self.is_mapped = false;
    }
}

impl fmt::Display for GpuBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpuBuffer(id={}, size={}, usage={}, mapped={})",
            self.id, self.size, self.usage, self.is_mapped,
        )
    }
}

// ── BufferPool ──────────────────────────────────────────────────────────────

/// Pool of pre-allocated GPU buffers for fast allocation/reuse.
#[derive(Debug)]
pub struct BufferPool {
    config: BufferConfig,
    /// Free buffers keyed by aligned size for O(1) lookup.
    free_list: HashMap<usize, Vec<GpuBuffer>>,
    /// Currently allocated (handed out) buffer IDs → size.
    allocated: HashMap<u64, usize>,
    /// Total bytes currently in the free list.
    free_bytes: usize,
    /// Total bytes currently allocated (in use).
    allocated_bytes: usize,
    /// High-water mark.
    peak_allocated_bytes: usize,
}

impl BufferPool {
    /// Create a new pool with the given configuration.
    #[must_use]
    pub fn new(config: BufferConfig) -> Self {
        Self {
            config,
            free_list: HashMap::new(),
            allocated: HashMap::new(),
            free_bytes: 0,
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
        }
    }

    /// Allocate a buffer. Reuses a pooled buffer when possible.
    #[must_use]
    pub fn alloc(&mut self, size: usize, usage: BufferUsage) -> GpuBuffer {
        let aligned = self.config.align_up(size);

        // Try to reuse from free list.
        if self.config.enable_pooling
            && let Some(list) = self.free_list.get_mut(&aligned)
            && let Some(mut buf) = list.pop()
        {
            self.free_bytes -= aligned;
            buf.usage = usage;
            buf.is_mapped = false;
            self.allocated.insert(buf.id, aligned);
            self.allocated_bytes += aligned;
            self.update_peak();
            return buf;
        }

        // Fresh allocation.
        let buf = GpuBuffer::new(size, self.config.alignment, usage);
        self.allocated.insert(buf.id, aligned);
        self.allocated_bytes += aligned;
        self.update_peak();
        buf
    }

    /// Return a buffer to the pool.
    ///
    /// Returns `true` if the buffer was successfully freed (was tracked).
    pub fn free(&mut self, buf: GpuBuffer) -> bool {
        let Some(size) = self.allocated.remove(&buf.id) else {
            return false;
        };
        self.allocated_bytes -= size;

        if self.config.enable_pooling && self.free_bytes + size <= self.config.pool_size_bytes {
            self.free_list.entry(size).or_default().push(buf);
            self.free_bytes += size;
        }
        // else: buffer is dropped (not pooled)
        true
    }

    /// Defragment: drop all free-list buffers.
    ///
    /// Returns the number of bytes released.
    pub fn defrag(&mut self) -> usize {
        let freed = self.free_bytes;
        self.free_list.clear();
        self.free_bytes = 0;
        freed
    }

    /// Pool statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            free_buffers: self.free_list.values().map(Vec::len).sum(),
            free_bytes: self.free_bytes,
            allocated_buffers: self.allocated.len(),
            allocated_bytes: self.allocated_bytes,
            peak_allocated_bytes: self.peak_allocated_bytes,
            pool_capacity_bytes: self.config.pool_size_bytes,
        }
    }

    const fn update_peak(&mut self) {
        if self.allocated_bytes > self.peak_allocated_bytes {
            self.peak_allocated_bytes = self.allocated_bytes;
        }
    }
}

/// Snapshot of pool statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolStats {
    pub free_buffers: usize,
    pub free_bytes: usize,
    pub allocated_buffers: usize,
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub pool_capacity_bytes: usize,
}

impl PoolStats {
    /// Utilisation ratio (allocated / capacity), 0.0–1.0+.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.pool_capacity_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.pool_capacity_bytes as f64
    }
}

// ── PinnedBuffer ────────────────────────────────────────────────────────────

/// Host-pinned (page-locked) buffer for fast DMA host↔device transfers.
#[derive(Debug)]
pub struct PinnedBuffer {
    /// Unique identifier.
    pub id: u64,
    /// Size in bytes.
    pub size: usize,
    /// Simulated host-side storage.
    data: Vec<u8>,
    /// Whether the buffer is currently locked.
    pub is_locked: bool,
}

impl PinnedBuffer {
    /// Allocate a new pinned buffer.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self { id: next_id(), size, data: vec![0u8; size], is_locked: true }
    }

    /// Write `src` into the pinned buffer starting at `offset`.
    ///
    /// Returns the number of bytes actually written.
    pub fn write(&mut self, offset: usize, src: &[u8]) -> usize {
        if offset >= self.size {
            return 0;
        }
        let end = (offset + src.len()).min(self.size);
        let count = end - offset;
        self.data[offset..end].copy_from_slice(&src[..count]);
        count
    }

    /// Read into `dst` from `offset`.
    ///
    /// Returns the number of bytes actually read.
    pub fn read(&self, offset: usize, dst: &mut [u8]) -> usize {
        if offset >= self.size {
            return 0;
        }
        let end = (offset + dst.len()).min(self.size);
        let count = end - offset;
        dst[..count].copy_from_slice(&self.data[offset..end]);
        count
    }

    /// Return raw slice reference.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Unlock (un-pin) the buffer.
    pub const fn unlock(&mut self) {
        self.is_locked = false;
    }
}

// ── StagingBuffer ───────────────────────────────────────────────────────────

/// Direction of a staging transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host → Device.
    HostToDevice,
    /// Device → Host.
    DeviceToHost,
}

impl fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H2D"),
            Self::DeviceToHost => write!(f, "D2H"),
        }
    }
}

/// Staging buffer for asynchronous host↔device transfers.
#[derive(Debug)]
pub struct StagingBuffer {
    /// Underlying pinned buffer.
    pub pinned: PinnedBuffer,
    /// Transfer direction this staging buffer is configured for.
    pub direction: TransferDirection,
    /// Whether a transfer is currently in flight.
    pub in_flight: bool,
}

impl StagingBuffer {
    /// Create a staging buffer for the given direction.
    #[must_use]
    pub fn new(size: usize, direction: TransferDirection) -> Self {
        Self { pinned: PinnedBuffer::new(size), direction, in_flight: false }
    }

    /// Begin a transfer (marks in-flight).
    pub const fn begin_transfer(&mut self) {
        self.in_flight = true;
    }

    /// Complete a transfer (clears in-flight).
    pub const fn complete_transfer(&mut self) {
        self.in_flight = false;
    }

    /// Size of the underlying pinned buffer.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.pinned.size
    }
}

// ── BufferView ──────────────────────────────────────────────────────────────

/// A typed view (sub-range) into an existing [`GpuBuffer`].
#[derive(Debug, Clone)]
pub struct BufferView {
    /// ID of the parent buffer.
    pub parent_id: u64,
    /// Byte offset into the parent.
    pub offset: usize,
    /// Size of this view in bytes.
    pub size: usize,
}

impl BufferView {
    /// Create a view into `parent` at `[offset..offset+size)`.
    ///
    /// Returns `None` if the range exceeds the parent bounds.
    #[must_use]
    pub const fn new(parent: &GpuBuffer, offset: usize, size: usize) -> Option<Self> {
        if offset + size > parent.size {
            return None;
        }
        Some(Self { parent_id: parent.id, offset, size })
    }

    /// End byte offset (exclusive).
    #[must_use]
    pub const fn end(&self) -> usize {
        self.offset + self.size
    }

    /// Whether two views overlap.
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        self.parent_id == other.parent_id && self.offset < other.end() && other.offset < self.end()
    }
}

// ── TransferRequest ─────────────────────────────────────────────────────────

/// Status of an enqueued transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    /// Waiting in the queue.
    Pending,
    /// Currently being executed.
    InProgress,
    /// Successfully completed.
    Completed,
    /// Failed.
    Failed,
}

/// A single host↔device copy request.
#[derive(Debug)]
pub struct TransferRequest {
    pub id: u64,
    pub direction: TransferDirection,
    pub size_bytes: usize,
    pub status: TransferStatus,
}

// ── TransferManager ─────────────────────────────────────────────────────────

/// Manages host↔device transfer scheduling.
#[derive(Debug)]
pub struct TransferManager {
    queue: Vec<TransferRequest>,
    completed_bytes: u64,
    total_transfers: u64,
}

impl TransferManager {
    /// Create a new transfer manager.
    #[must_use]
    pub const fn new() -> Self {
        Self { queue: Vec::new(), completed_bytes: 0, total_transfers: 0 }
    }

    /// Enqueue a copy request. Returns the transfer ID.
    pub fn enqueue_copy(&mut self, direction: TransferDirection, size_bytes: usize) -> u64 {
        let id = next_id();
        self.queue.push(TransferRequest {
            id,
            direction,
            size_bytes,
            status: TransferStatus::Pending,
        });
        id
    }

    /// Execute all pending transfers synchronously.
    ///
    /// Returns the number of transfers completed in this call.
    pub fn sync(&mut self) -> usize {
        let mut count = 0;
        for req in &mut self.queue {
            if req.status == TransferStatus::Pending {
                req.status = TransferStatus::InProgress;
                // simulate instant completion
                req.status = TransferStatus::Completed;
                self.completed_bytes += req.size_bytes as u64;
                self.total_transfers += 1;
                count += 1;
            }
        }
        count
    }

    /// Number of requests still pending.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.iter().filter(|r| r.status == TransferStatus::Pending).count()
    }

    /// Total bytes transferred (completed).
    #[must_use]
    pub const fn completed_bytes(&self) -> u64 {
        self.completed_bytes
    }

    /// Total number of transfers completed.
    #[must_use]
    pub const fn total_transfers(&self) -> u64 {
        self.total_transfers
    }

    /// Drain completed requests from the queue, returning them.
    pub fn drain_completed(&mut self) -> Vec<TransferRequest> {
        let mut completed = Vec::new();
        self.queue.retain(|r| {
            if r.status == TransferStatus::Completed {
                false // remove from queue (will be moved)
            } else {
                true
            }
        });
        // Re-do: we need to actually move them out.
        // Use partition approach:
        let mut kept = Vec::new();
        for req in self.queue.drain(..) {
            if req.status == TransferStatus::Completed {
                completed.push(req);
            } else {
                kept.push(req);
            }
        }
        self.queue = kept;
        completed
    }
}

impl Default for TransferManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── BufferLifetimeTracker ───────────────────────────────────────────────────

/// Tracks buffer lifetimes for automatic cleanup.
#[derive(Debug)]
pub struct BufferLifetimeTracker {
    /// Buffer ID → creation instant.
    tracked: HashMap<u64, Instant>,
}

impl BufferLifetimeTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self { tracked: HashMap::new() }
    }

    /// Start tracking a buffer.
    pub fn track(&mut self, buffer_id: u64) {
        self.tracked.insert(buffer_id, Instant::now());
    }

    /// Stop tracking a buffer. Returns `true` if it was tracked.
    pub fn release(&mut self, buffer_id: u64) -> bool {
        self.tracked.remove(&buffer_id).is_some()
    }

    /// Number of buffers currently tracked.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.tracked.len()
    }

    /// Whether a specific buffer is tracked.
    #[must_use]
    pub fn is_tracked(&self, buffer_id: u64) -> bool {
        self.tracked.contains_key(&buffer_id)
    }

    /// Return IDs of all tracked buffers.
    #[must_use]
    pub fn tracked_ids(&self) -> Vec<u64> {
        self.tracked.keys().copied().collect()
    }
}

impl Default for BufferLifetimeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── BufferMetrics ───────────────────────────────────────────────────────────

/// Aggregate metrics across the buffer subsystem.
#[derive(Debug, Clone)]
pub struct BufferMetrics {
    /// Total bytes allocated (active).
    pub total_allocated: u64,
    /// Pool utilisation ratio (0.0–1.0+).
    pub pool_utilization: f64,
    /// Cumulative bytes transferred.
    pub transfer_bandwidth_bytes: u64,
    /// Peak allocated bytes observed.
    pub peak_usage: u64,
    /// Number of pool hits (reuses).
    pub pool_hits: u64,
    /// Number of pool misses (new allocations).
    pub pool_misses: u64,
}

impl BufferMetrics {
    /// Create zeroed metrics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total_allocated: 0,
            pool_utilization: 0.0,
            transfer_bandwidth_bytes: 0,
            peak_usage: 0,
            pool_hits: 0,
            pool_misses: 0,
        }
    }

    /// Build metrics from pool stats and transfer manager.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_components(pool_stats: &PoolStats, transfer_mgr: &TransferManager) -> Self {
        Self {
            total_allocated: pool_stats.allocated_bytes as u64,
            pool_utilization: pool_stats.utilization(),
            transfer_bandwidth_bytes: transfer_mgr.completed_bytes(),
            peak_usage: pool_stats.peak_allocated_bytes as u64,
            pool_hits: 0,
            pool_misses: 0,
        }
    }
}

impl Default for BufferMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BufferMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BufferMetrics(alloc={}B, peak={}B, pool={:.1}%, xfer={}B)",
            self.total_allocated,
            self.peak_usage,
            self.pool_utilization * 100.0,
            self.transfer_bandwidth_bytes,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── BufferConfig ────────────────────────────────────────────────────

    #[test]
    fn config_default_alignment() {
        let cfg = BufferConfig::default();
        assert_eq!(cfg.alignment, 256);
    }

    #[test]
    fn config_default_pooling_enabled() {
        let cfg = BufferConfig::default();
        assert!(cfg.enable_pooling);
        assert!(!cfg.use_pinned);
    }

    #[test]
    fn config_default_pool_size() {
        let cfg = BufferConfig::default();
        assert_eq!(cfg.pool_size_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn config_align_up_exact() {
        let cfg = BufferConfig { alignment: 256, ..Default::default() };
        assert_eq!(cfg.align_up(256), 256);
    }

    #[test]
    fn config_align_up_rounds() {
        let cfg = BufferConfig { alignment: 256, ..Default::default() };
        assert_eq!(cfg.align_up(100), 256);
        assert_eq!(cfg.align_up(257), 512);
    }

    #[test]
    fn config_align_up_zero() {
        let cfg = BufferConfig { alignment: 256, ..Default::default() };
        assert_eq!(cfg.align_up(0), 0);
    }

    #[test]
    fn config_align_up_power_of_two() {
        let cfg = BufferConfig { alignment: 64, ..Default::default() };
        assert_eq!(cfg.align_up(1), 64);
        assert_eq!(cfg.align_up(64), 64);
        assert_eq!(cfg.align_up(65), 128);
    }

    // ── BufferUsage ─────────────────────────────────────────────────────

    #[test]
    fn buffer_usage_display() {
        assert_eq!(BufferUsage::ReadOnly.to_string(), "ReadOnly");
        assert_eq!(BufferUsage::WriteOnly.to_string(), "WriteOnly");
        assert_eq!(BufferUsage::ReadWrite.to_string(), "ReadWrite");
        assert_eq!(BufferUsage::Kernel.to_string(), "Kernel");
        assert_eq!(BufferUsage::Transfer.to_string(), "Transfer");
        assert_eq!(BufferUsage::Staging.to_string(), "Staging");
    }

    #[test]
    fn buffer_usage_equality() {
        assert_eq!(BufferUsage::Kernel, BufferUsage::Kernel);
        assert_ne!(BufferUsage::ReadOnly, BufferUsage::WriteOnly);
    }

    #[test]
    fn buffer_usage_copy() {
        let a = BufferUsage::Transfer;
        let b = a;
        assert_eq!(a, b);
    }

    // ── GpuBuffer ───────────────────────────────────────────────────────

    #[test]
    fn gpu_buffer_new_aligns_size() {
        let buf = GpuBuffer::new(100, 256, BufferUsage::ReadOnly);
        assert_eq!(buf.size, 256);
        assert_eq!(buf.alignment, 256);
    }

    #[test]
    fn gpu_buffer_exact_alignment() {
        let buf = GpuBuffer::new(512, 256, BufferUsage::ReadWrite);
        assert_eq!(buf.size, 512);
    }

    #[test]
    fn gpu_buffer_unique_ids() {
        let a = GpuBuffer::new(64, 64, BufferUsage::Kernel);
        let b = GpuBuffer::new(64, 64, BufferUsage::Kernel);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn gpu_buffer_not_mapped_initially() {
        let buf = GpuBuffer::new(128, 64, BufferUsage::ReadOnly);
        assert!(!buf.is_mapped);
    }

    #[test]
    fn gpu_buffer_map_unmap() {
        let mut buf = GpuBuffer::new(128, 64, BufferUsage::ReadWrite);
        buf.map();
        assert!(buf.is_mapped);
        buf.unmap();
        assert!(!buf.is_mapped);
    }

    #[test]
    fn gpu_buffer_with_config() {
        let cfg = BufferConfig { alignment: 512, ..Default::default() };
        let buf = GpuBuffer::with_config(100, BufferUsage::Kernel, &cfg);
        assert_eq!(buf.size, 512);
        assert_eq!(buf.alignment, 512);
    }

    #[test]
    fn gpu_buffer_display() {
        let buf = GpuBuffer::new(256, 256, BufferUsage::Transfer);
        let s = buf.to_string();
        assert!(s.contains("GpuBuffer"));
        assert!(s.contains("Transfer"));
    }

    #[test]
    fn gpu_buffer_device_ptr_offset_default() {
        let buf = GpuBuffer::new(64, 64, BufferUsage::ReadOnly);
        assert_eq!(buf.device_ptr_offset, 0);
    }

    #[test]
    fn gpu_buffer_clone() {
        let buf = GpuBuffer::new(64, 64, BufferUsage::ReadOnly);
        let buf2 = buf.clone();
        assert_eq!(buf.id, buf2.id);
        assert_eq!(buf.size, buf2.size);
    }

    // ── BufferPool — allocation ─────────────────────────────────────────

    #[test]
    fn pool_alloc_basic() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let buf = pool.alloc(100, BufferUsage::ReadOnly);
        assert_eq!(buf.size, 256); // aligned
        assert_eq!(pool.stats().allocated_buffers, 1);
    }

    #[test]
    fn pool_alloc_multiple() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let _a = pool.alloc(100, BufferUsage::ReadOnly);
        let _b = pool.alloc(200, BufferUsage::WriteOnly);
        assert_eq!(pool.stats().allocated_buffers, 2);
        assert_eq!(pool.stats().allocated_bytes, 512); // 256 + 256
    }

    #[test]
    fn pool_free_returns_true() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let buf = pool.alloc(100, BufferUsage::ReadOnly);
        assert!(pool.free(buf));
    }

    #[test]
    fn pool_free_unknown_returns_false() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let buf = GpuBuffer::new(64, 64, BufferUsage::ReadOnly);
        assert!(!pool.free(buf));
    }

    #[test]
    fn pool_reuse_after_free() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let buf = pool.alloc(100, BufferUsage::ReadOnly);
        let original_id = buf.id;
        pool.free(buf);

        // Same size should reuse.
        let buf2 = pool.alloc(100, BufferUsage::WriteOnly);
        assert_eq!(buf2.id, original_id);
        assert_eq!(buf2.usage, BufferUsage::WriteOnly);
    }

    #[test]
    fn pool_no_reuse_when_disabled() {
        let cfg = BufferConfig { enable_pooling: false, ..Default::default() };
        let mut pool = BufferPool::new(cfg);
        let buf = pool.alloc(100, BufferUsage::ReadOnly);
        let original_id = buf.id;
        pool.free(buf);

        let buf2 = pool.alloc(100, BufferUsage::ReadOnly);
        assert_ne!(buf2.id, original_id);
    }

    #[test]
    fn pool_stats_after_alloc_free() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let buf = pool.alloc(100, BufferUsage::ReadOnly);
        assert_eq!(pool.stats().allocated_bytes, 256);
        pool.free(buf);
        assert_eq!(pool.stats().allocated_bytes, 0);
        assert_eq!(pool.stats().free_bytes, 256);
        assert_eq!(pool.stats().free_buffers, 1);
    }

    #[test]
    fn pool_peak_tracking() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let a = pool.alloc(256, BufferUsage::ReadOnly);
        let b = pool.alloc(256, BufferUsage::ReadOnly);
        assert_eq!(pool.stats().peak_allocated_bytes, 512);
        pool.free(a);
        pool.free(b);
        assert_eq!(pool.stats().peak_allocated_bytes, 512); // unchanged
    }

    // ── BufferPool — defragmentation ────────────────────────────────────

    #[test]
    fn pool_defrag_clears_free_list() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let buf = pool.alloc(100, BufferUsage::ReadOnly);
        pool.free(buf);
        assert_eq!(pool.stats().free_buffers, 1);

        let freed = pool.defrag();
        assert_eq!(freed, 256);
        assert_eq!(pool.stats().free_buffers, 0);
        assert_eq!(pool.stats().free_bytes, 0);
    }

    #[test]
    fn pool_defrag_empty() {
        let mut pool = BufferPool::new(BufferConfig::default());
        assert_eq!(pool.defrag(), 0);
    }

    // ── BufferPool — capacity limit ─────────────────────────────────────

    #[test]
    fn pool_capacity_limit_drops_excess() {
        let cfg = BufferConfig { pool_size_bytes: 256, alignment: 256, ..Default::default() };
        let mut pool = BufferPool::new(cfg);
        let a = pool.alloc(256, BufferUsage::ReadOnly);
        let b = pool.alloc(256, BufferUsage::ReadOnly);
        pool.free(a); // goes to pool (256 ≤ 256)
        pool.free(b); // dropped: would exceed pool limit
        assert_eq!(pool.stats().free_bytes, 256);
    }

    // ── PoolStats ───────────────────────────────────────────────────────

    #[test]
    fn pool_stats_utilization_zero() {
        let stats = PoolStats {
            free_buffers: 0,
            free_bytes: 0,
            allocated_buffers: 0,
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
            pool_capacity_bytes: 1024,
        };
        assert!((stats.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pool_stats_utilization_half() {
        let stats = PoolStats {
            free_buffers: 0,
            free_bytes: 0,
            allocated_buffers: 1,
            allocated_bytes: 512,
            peak_allocated_bytes: 512,
            pool_capacity_bytes: 1024,
        };
        assert!((stats.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn pool_stats_utilization_zero_capacity() {
        let stats = PoolStats {
            free_buffers: 0,
            free_bytes: 0,
            allocated_buffers: 0,
            allocated_bytes: 0,
            peak_allocated_bytes: 0,
            pool_capacity_bytes: 0,
        };
        assert!((stats.utilization() - 0.0).abs() < f64::EPSILON);
    }

    // ── PinnedBuffer ────────────────────────────────────────────────────

    #[test]
    fn pinned_buffer_new() {
        let pb = PinnedBuffer::new(1024);
        assert_eq!(pb.size, 1024);
        assert!(pb.is_locked);
        assert_eq!(pb.as_slice().len(), 1024);
    }

    #[test]
    fn pinned_buffer_write_read() {
        let mut pb = PinnedBuffer::new(256);
        let data = [1u8, 2, 3, 4];
        let written = pb.write(0, &data);
        assert_eq!(written, 4);

        let mut out = [0u8; 4];
        let read = pb.read(0, &mut out);
        assert_eq!(read, 4);
        assert_eq!(out, [1, 2, 3, 4]);
    }

    #[test]
    fn pinned_buffer_write_at_offset() {
        let mut pb = PinnedBuffer::new(256);
        let data = [0xAA, 0xBB];
        pb.write(10, &data);

        let mut out = [0u8; 2];
        pb.read(10, &mut out);
        assert_eq!(out, [0xAA, 0xBB]);
    }

    #[test]
    fn pinned_buffer_write_clipped() {
        let mut pb = PinnedBuffer::new(4);
        let data = [1, 2, 3, 4, 5, 6]; // 6 bytes, only 4 fit
        let written = pb.write(0, &data);
        assert_eq!(written, 4);
    }

    #[test]
    fn pinned_buffer_write_past_end() {
        let mut pb = PinnedBuffer::new(4);
        let written = pb.write(100, &[1, 2]);
        assert_eq!(written, 0);
    }

    #[test]
    fn pinned_buffer_read_past_end() {
        let pb = PinnedBuffer::new(4);
        let mut out = [0u8; 2];
        let read = pb.read(100, &mut out);
        assert_eq!(read, 0);
    }

    #[test]
    fn pinned_buffer_unlock() {
        let mut pb = PinnedBuffer::new(64);
        assert!(pb.is_locked);
        pb.unlock();
        assert!(!pb.is_locked);
    }

    #[test]
    fn pinned_buffer_zeroed() {
        let pb = PinnedBuffer::new(16);
        assert!(pb.as_slice().iter().all(|&b| b == 0));
    }

    // ── StagingBuffer ───────────────────────────────────────────────────

    #[test]
    fn staging_buffer_h2d() {
        let sb = StagingBuffer::new(512, TransferDirection::HostToDevice);
        assert_eq!(sb.direction, TransferDirection::HostToDevice);
        assert_eq!(sb.size(), 512);
        assert!(!sb.in_flight);
    }

    #[test]
    fn staging_buffer_d2h() {
        let sb = StagingBuffer::new(256, TransferDirection::DeviceToHost);
        assert_eq!(sb.direction, TransferDirection::DeviceToHost);
    }

    #[test]
    fn staging_buffer_transfer_lifecycle() {
        let mut sb = StagingBuffer::new(128, TransferDirection::HostToDevice);
        assert!(!sb.in_flight);
        sb.begin_transfer();
        assert!(sb.in_flight);
        sb.complete_transfer();
        assert!(!sb.in_flight);
    }

    #[test]
    fn staging_buffer_pinned_write_read() {
        let mut sb = StagingBuffer::new(64, TransferDirection::HostToDevice);
        sb.pinned.write(0, &[42, 43]);
        let mut out = [0u8; 2];
        sb.pinned.read(0, &mut out);
        assert_eq!(out, [42, 43]);
    }

    #[test]
    fn transfer_direction_display() {
        assert_eq!(TransferDirection::HostToDevice.to_string(), "H2D");
        assert_eq!(TransferDirection::DeviceToHost.to_string(), "D2H");
    }

    // ── BufferView ──────────────────────────────────────────────────────

    #[test]
    fn buffer_view_valid() {
        let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let view = BufferView::new(&buf, 0, 512);
        assert!(view.is_some());
        let v = view.unwrap();
        assert_eq!(v.offset, 0);
        assert_eq!(v.size, 512);
        assert_eq!(v.parent_id, buf.id);
    }

    #[test]
    fn buffer_view_full_range() {
        let buf = GpuBuffer::new(256, 256, BufferUsage::ReadOnly);
        let view = BufferView::new(&buf, 0, 256);
        assert!(view.is_some());
    }

    #[test]
    fn buffer_view_out_of_bounds() {
        let buf = GpuBuffer::new(256, 256, BufferUsage::ReadOnly);
        assert!(BufferView::new(&buf, 200, 100).is_none());
    }

    #[test]
    fn buffer_view_zero_size() {
        let buf = GpuBuffer::new(256, 256, BufferUsage::ReadOnly);
        let view = BufferView::new(&buf, 128, 0);
        assert!(view.is_some());
        assert_eq!(view.unwrap().size, 0);
    }

    #[test]
    fn buffer_view_end() {
        let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let v = BufferView::new(&buf, 100, 200).unwrap();
        assert_eq!(v.end(), 300);
    }

    #[test]
    fn buffer_view_overlap_yes() {
        let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let a = BufferView::new(&buf, 0, 200).unwrap();
        let b = BufferView::new(&buf, 100, 200).unwrap();
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn buffer_view_overlap_no() {
        let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let a = BufferView::new(&buf, 0, 100).unwrap();
        let b = BufferView::new(&buf, 100, 100).unwrap();
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn buffer_view_overlap_different_parents() {
        let buf1 = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let buf2 = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let a = BufferView::new(&buf1, 0, 200).unwrap();
        let b = BufferView::new(&buf2, 0, 200).unwrap();
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn buffer_view_clone() {
        let buf = GpuBuffer::new(512, 256, BufferUsage::ReadWrite);
        let v = BufferView::new(&buf, 0, 256).unwrap();
        let v2 = v.clone();
        assert_eq!(v.parent_id, v2.parent_id);
        assert_eq!(v.offset, v2.offset);
    }

    // ── TransferManager ─────────────────────────────────────────────────

    #[test]
    fn transfer_manager_enqueue() {
        let mut tm = TransferManager::new();
        let id = tm.enqueue_copy(TransferDirection::HostToDevice, 1024);
        assert!(id > 0);
        assert_eq!(tm.pending_count(), 1);
    }

    #[test]
    fn transfer_manager_sync() {
        let mut tm = TransferManager::new();
        tm.enqueue_copy(TransferDirection::HostToDevice, 1024);
        tm.enqueue_copy(TransferDirection::DeviceToHost, 512);
        let completed = tm.sync();
        assert_eq!(completed, 2);
        assert_eq!(tm.pending_count(), 0);
        assert_eq!(tm.completed_bytes(), 1536);
    }

    #[test]
    fn transfer_manager_sync_empty() {
        let mut tm = TransferManager::new();
        assert_eq!(tm.sync(), 0);
    }

    #[test]
    fn transfer_manager_total_transfers() {
        let mut tm = TransferManager::new();
        tm.enqueue_copy(TransferDirection::HostToDevice, 100);
        tm.enqueue_copy(TransferDirection::HostToDevice, 200);
        tm.sync();
        assert_eq!(tm.total_transfers(), 2);
    }

    #[test]
    fn transfer_manager_pending_after_partial() {
        let mut tm = TransferManager::new();
        tm.enqueue_copy(TransferDirection::HostToDevice, 100);
        tm.sync();
        tm.enqueue_copy(TransferDirection::DeviceToHost, 200);
        assert_eq!(tm.pending_count(), 1);
    }

    #[test]
    fn transfer_manager_default() {
        let tm = TransferManager::default();
        assert_eq!(tm.pending_count(), 0);
        assert_eq!(tm.completed_bytes(), 0);
    }

    #[test]
    fn transfer_manager_cumulative_bytes() {
        let mut tm = TransferManager::new();
        tm.enqueue_copy(TransferDirection::HostToDevice, 100);
        tm.sync();
        tm.enqueue_copy(TransferDirection::HostToDevice, 200);
        tm.sync();
        assert_eq!(tm.completed_bytes(), 300);
    }

    // ── BufferLifetimeTracker ───────────────────────────────────────────

    #[test]
    fn lifetime_tracker_track_release() {
        let mut tracker = BufferLifetimeTracker::new();
        tracker.track(1);
        assert_eq!(tracker.active_count(), 1);
        assert!(tracker.is_tracked(1));

        assert!(tracker.release(1));
        assert_eq!(tracker.active_count(), 0);
        assert!(!tracker.is_tracked(1));
    }

    #[test]
    fn lifetime_tracker_release_unknown() {
        let mut tracker = BufferLifetimeTracker::new();
        assert!(!tracker.release(999));
    }

    #[test]
    fn lifetime_tracker_multiple() {
        let mut tracker = BufferLifetimeTracker::new();
        tracker.track(1);
        tracker.track(2);
        tracker.track(3);
        assert_eq!(tracker.active_count(), 3);

        let mut ids = tracker.tracked_ids();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn lifetime_tracker_default() {
        let tracker = BufferLifetimeTracker::default();
        assert_eq!(tracker.active_count(), 0);
    }

    #[test]
    fn lifetime_tracker_double_track_overwrites() {
        let mut tracker = BufferLifetimeTracker::new();
        tracker.track(1);
        tracker.track(1); // overwrites
        assert_eq!(tracker.active_count(), 1);
    }

    // ── BufferMetrics ───────────────────────────────────────────────────

    #[test]
    fn metrics_new_zeroed() {
        let m = BufferMetrics::new();
        assert_eq!(m.total_allocated, 0);
        assert_eq!(m.peak_usage, 0);
        assert!((m.pool_utilization - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_default_equals_new() {
        let a = BufferMetrics::new();
        let b = BufferMetrics::default();
        assert_eq!(a.total_allocated, b.total_allocated);
    }

    #[test]
    fn metrics_from_components() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let _buf = pool.alloc(256, BufferUsage::ReadOnly);
        let mut tm = TransferManager::new();
        tm.enqueue_copy(TransferDirection::HostToDevice, 1024);
        tm.sync();

        let m = BufferMetrics::from_components(&pool.stats(), &tm);
        assert_eq!(m.total_allocated, 256);
        assert_eq!(m.transfer_bandwidth_bytes, 1024);
    }

    #[test]
    fn metrics_display() {
        let m = BufferMetrics {
            total_allocated: 1024,
            pool_utilization: 0.5,
            transfer_bandwidth_bytes: 2048,
            peak_usage: 1024,
            pool_hits: 0,
            pool_misses: 0,
        };
        let s = m.to_string();
        assert!(s.contains("1024B"));
        assert!(s.contains("50.0%"));
    }

    #[test]
    fn metrics_clone() {
        let m = BufferMetrics::new();
        let m2 = m.clone();
        assert_eq!(m.total_allocated, m2.total_allocated);
    }

    // ── Integration / edge cases ────────────────────────────────────────

    #[test]
    fn pool_alloc_free_alloc_reuse_cycle() {
        let mut pool = BufferPool::new(BufferConfig::default());
        for _ in 0..10 {
            let buf = pool.alloc(128, BufferUsage::Kernel);
            pool.free(buf);
        }
        // Should have exactly 1 buffer in free list (all reused).
        assert_eq!(pool.stats().free_buffers, 1);
        assert_eq!(pool.stats().allocated_buffers, 0);
    }

    #[test]
    fn staging_to_transfer_pipeline() {
        let mut staging = StagingBuffer::new(256, TransferDirection::HostToDevice);
        staging.pinned.write(0, &[1, 2, 3, 4]);
        staging.begin_transfer();

        let mut tm = TransferManager::new();
        tm.enqueue_copy(TransferDirection::HostToDevice, 4);
        tm.sync();

        staging.complete_transfer();
        assert!(!staging.in_flight);
        assert_eq!(tm.completed_bytes(), 4);
    }

    #[test]
    fn lifetime_tracker_with_pool() {
        let mut pool = BufferPool::new(BufferConfig::default());
        let mut tracker = BufferLifetimeTracker::new();

        let buf = pool.alloc(128, BufferUsage::ReadOnly);
        tracker.track(buf.id);
        assert!(tracker.is_tracked(buf.id));

        tracker.release(buf.id);
        pool.free(buf);
        assert_eq!(tracker.active_count(), 0);
        assert_eq!(pool.stats().allocated_buffers, 0);
    }

    #[test]
    fn multiple_views_same_buffer() {
        let buf = GpuBuffer::new(1024, 256, BufferUsage::ReadWrite);
        let v1 = BufferView::new(&buf, 0, 256).unwrap();
        let v2 = BufferView::new(&buf, 256, 256).unwrap();
        let v3 = BufferView::new(&buf, 512, 512).unwrap();
        assert!(!v1.overlaps(&v2));
        assert!(!v2.overlaps(&v3));
        assert!(!v1.overlaps(&v3));
    }

    #[test]
    fn transfer_status_enum_values() {
        assert_eq!(TransferStatus::Pending, TransferStatus::Pending);
        assert_ne!(TransferStatus::Pending, TransferStatus::Completed);
        assert_ne!(TransferStatus::InProgress, TransferStatus::Failed);
    }

    #[test]
    fn pool_many_different_sizes() {
        let cfg = BufferConfig { alignment: 64, ..Default::default() };
        let mut pool = BufferPool::new(cfg);
        let mut bufs = Vec::new();
        for i in 1..=20 {
            bufs.push(pool.alloc(i * 64, BufferUsage::ReadWrite));
        }
        assert_eq!(pool.stats().allocated_buffers, 20);
        for buf in bufs {
            pool.free(buf);
        }
        assert_eq!(pool.stats().allocated_buffers, 0);
        assert_eq!(pool.stats().free_buffers, 20);
    }

    #[test]
    fn pinned_buffer_full_roundtrip() {
        let mut pb = PinnedBuffer::new(8);
        let data = [10, 20, 30, 40, 50, 60, 70, 80];
        pb.write(0, &data);
        let mut out = [0u8; 8];
        pb.read(0, &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn gpu_buffer_alignment_one() {
        let buf = GpuBuffer::new(7, 1, BufferUsage::ReadOnly);
        assert_eq!(buf.size, 7);
    }
}
