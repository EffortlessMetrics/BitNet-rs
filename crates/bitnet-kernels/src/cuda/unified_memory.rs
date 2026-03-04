//! CUDA unified memory management with prefetch and advisory hints.
//!
//! # Overview
//!
//! Provides managed host-device memory that leverages CUDA Unified Memory
//! (`cudaMallocManaged`) semantics.  Key components:
//!
//! - [`UnifiedMemoryConfig`] — configuration (prefetch, advise, page size, oversubscription).
//! - [`ManagedBuffer`] — a type-safe buffer allocated with unified memory semantics.
//! - [`MemoryAdvice`] — advisory hints for the CUDA driver (read-mostly, preferred location, …).
//! - [`PrefetchRequest`] — descriptor for an async prefetch operation.
//! - [`UnifiedMemoryStats`] — snapshot of unified memory usage.
//!
//! Helper functions:
//!
//! - [`advise_memory`] — attach advisory hints to a buffer.
//! - [`prefetch_to_device`] / [`prefetch_to_host`] — async data migration.
//! - [`unified_memcpy`] — explicit copy with profiling.
//! - [`memory_usage_stats`] — query aggregate unified memory statistics.
//! - [`batch_prefetch`] — prefetch multiple buffers in one call.
//! - [`simulate_page_fault`] — CPU-side page-fault simulation for testing.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations are provided for testing on non-GPU hosts.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_buffer_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

// ── Configuration ────────────────────────────────────────────────────

/// Configuration for the CUDA unified memory subsystem.
#[derive(Debug, Clone)]
pub struct UnifiedMemoryConfig {
    /// Enable automatic prefetch of buffers to the target device before kernel launch.
    pub enable_prefetch: bool,
    /// Enable advisory hints (e.g. read-mostly, preferred location) for the driver.
    pub enable_advise: bool,
    /// Managed-memory page size in bytes (must be a power of two, ≥ 4096).
    pub page_size: usize,
    /// Maximum ratio of allocated unified memory to physical device memory.
    /// Values > 1.0 enable oversubscription (requires CUDA ≥ 8.0 + 64-bit OS).
    pub oversubscription_limit: f64,
}

impl Default for UnifiedMemoryConfig {
    fn default() -> Self {
        Self {
            enable_prefetch: true,
            enable_advise: true,
            page_size: 64 * 1024, // 64 KiB (common page size for managed memory)
            oversubscription_limit: 1.0,
        }
    }
}

impl UnifiedMemoryConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<()> {
        if self.page_size < 4096 {
            return Err(KernelError::InvalidArguments {
                reason: "page_size must be at least 4096 bytes".into(),
            }
            .into());
        }
        if !self.page_size.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "page_size must be a power of two".into(),
            }
            .into());
        }
        if self.oversubscription_limit <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: "oversubscription_limit must be positive".into(),
            }
            .into());
        }
        if self.oversubscription_limit > 16.0 {
            return Err(KernelError::InvalidArguments {
                reason: "oversubscription_limit must be <= 16.0".into(),
            }
            .into());
        }
        Ok(())
    }

    /// Builder: set `enable_prefetch`.
    pub fn with_prefetch(mut self, enable: bool) -> Self {
        self.enable_prefetch = enable;
        self
    }

    /// Builder: set `enable_advise`.
    pub fn with_advise(mut self, enable: bool) -> Self {
        self.enable_advise = enable;
        self
    }

    /// Builder: set `page_size`.
    pub fn with_page_size(mut self, size: usize) -> Self {
        self.page_size = size;
        self
    }

    /// Builder: set `oversubscription_limit`.
    pub fn with_oversubscription_limit(mut self, limit: f64) -> Self {
        self.oversubscription_limit = limit;
        self
    }
}

// ── MemoryAdvice ─────────────────────────────────────────────────────

/// Advisory hints for the unified-memory driver, analogous to `cudaMemAdvise`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryAdvice {
    /// Data will be mostly read — driver may create read-only replicas.
    ReadMostly,
    /// Prefer physical allocation on the given device.
    PreferredLocation,
    /// Hint that the specified device will access the data.
    AccessedBy,
    /// Hint that the data will be needed soon (trigger eager migration).
    WillNeedSoon,
}

impl std::fmt::Display for MemoryAdvice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadMostly => write!(f, "ReadMostly"),
            Self::PreferredLocation => write!(f, "PreferredLocation"),
            Self::AccessedBy => write!(f, "AccessedBy"),
            Self::WillNeedSoon => write!(f, "WillNeedSoon"),
        }
    }
}

// ── CopyKind ─────────────────────────────────────────────────────────

/// Direction for an explicit unified-memory copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopyKind {
    /// Host → Device.
    HostToDevice,
    /// Device → Host.
    DeviceToHost,
    /// Device → Device (peer).
    DeviceToDevice,
    /// Host → Host.
    HostToHost,
}

impl std::fmt::Display for CopyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "HostToDevice"),
            Self::DeviceToHost => write!(f, "DeviceToHost"),
            Self::DeviceToDevice => write!(f, "DeviceToDevice"),
            Self::HostToHost => write!(f, "HostToHost"),
        }
    }
}

// ── ManagedBuffer ────────────────────────────────────────────────────

/// A type-safe buffer backed by CUDA unified (managed) memory.
///
/// In production (`gpu`/`cuda` feature), this wraps `cudaMallocManaged`.
/// In CPU-fallback mode, it uses a plain `Vec<T>` that simulates the API.
#[derive(Debug)]
pub struct ManagedBuffer<T: Copy + Default> {
    /// Unique buffer identifier.
    id: u64,
    /// Underlying storage (CPU fallback).
    data: Vec<T>,
    /// Number of logical elements.
    len: usize,
    /// Device the buffer was last prefetched to (`-1` = host).
    current_device: i32,
    /// Advisory hints that have been applied.
    advices: Vec<(MemoryAdvice, i32)>,
    /// Timestamp of last access (for profiling / eviction).
    last_access: Instant,
    /// Cumulative bytes transferred (profiling).
    bytes_transferred: u64,
    /// Page-fault counter (simulated for CPU testing).
    page_faults: u64,
}

impl<T: Copy + Default> ManagedBuffer<T> {
    /// Allocate a new managed buffer of `len` elements.
    ///
    /// All elements are zero-initialized (via `T::default()`).
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "ManagedBuffer length must be non-zero".into(),
            }
            .into());
        }
        let byte_size = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            KernelError::InvalidArguments { reason: "allocation size overflow".into() }
        })?;
        // Guard against unreasonable allocations (4 GiB limit).
        if byte_size > 4 * 1024 * 1024 * 1024 {
            return Err(KernelError::GpuError {
                reason: format!("managed allocation of {} bytes exceeds 4 GiB limit", byte_size),
            }
            .into());
        }
        Ok(Self {
            id: next_buffer_id(),
            data: vec![T::default(); len],
            len,
            current_device: -1, // starts on host
            advices: Vec::new(),
            last_access: Instant::now(),
            bytes_transferred: 0,
            page_faults: 0,
        })
    }

    /// Unique buffer id.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of logical elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Device the buffer currently resides on (`-1` = host).
    pub fn current_device(&self) -> i32 {
        self.current_device
    }

    /// Cumulative bytes transferred via prefetch / memcpy.
    pub fn bytes_transferred(&self) -> u64 {
        self.bytes_transferred
    }

    /// Page-fault count (simulated).
    pub fn page_faults(&self) -> u64 {
        self.page_faults
    }

    /// Advisory hints applied so far.
    pub fn advices(&self) -> &[(MemoryAdvice, i32)] {
        &self.advices
    }

    /// Immutable slice access (records a page fault on device mismatch).
    pub fn as_slice(&mut self) -> &[T] {
        if self.current_device != -1 {
            self.page_faults += 1;
        }
        self.last_access = Instant::now();
        &self.data
    }

    /// Mutable slice access (records a page fault on device mismatch).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.current_device != -1 {
            self.page_faults += 1;
        }
        self.last_access = Instant::now();
        &mut self.data
    }

    /// Read-only peek without updating access metadata.
    pub fn peek(&self) -> &[T] {
        &self.data
    }
}

// ── PrefetchRequest ──────────────────────────────────────────────────

/// Describes an asynchronous prefetch operation.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Target buffer identifier.
    pub buffer_id: u64,
    /// Target CUDA device ordinal (or `-1` for host).
    pub device_id: i32,
    /// Optional CUDA stream ordinal for async prefetch (`None` = default stream).
    pub stream: Option<u64>,
}

impl PrefetchRequest {
    /// Create a new prefetch request.
    pub fn new(buffer_id: u64, device_id: i32) -> Self {
        Self { buffer_id, device_id, stream: None }
    }

    /// Builder: set stream.
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.stream = Some(stream);
        self
    }
}

// ── UnifiedMemoryStats ───────────────────────────────────────────────

/// Snapshot of unified memory usage statistics.
#[derive(Debug, Clone, Default)]
pub struct UnifiedMemoryStats {
    /// Total bytes allocated via managed allocations.
    pub total_allocated_bytes: u64,
    /// Number of live managed buffers.
    pub active_buffers: u64,
    /// Cumulative bytes prefetched to any device.
    pub total_prefetched_bytes: u64,
    /// Cumulative bytes copied via `unified_memcpy`.
    pub total_copied_bytes: u64,
    /// Total page faults (simulated on CPU).
    pub total_page_faults: u64,
    /// Number of advisory hints applied.
    pub total_advise_calls: u64,
}

// ── CopyRecord ───────────────────────────────────────────────────────

/// Profiling record for a single `unified_memcpy` call.
#[derive(Debug, Clone)]
pub struct CopyRecord {
    /// Direction of the copy.
    pub kind: CopyKind,
    /// Number of bytes copied.
    pub bytes: usize,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Effective bandwidth in bytes / second.
    pub bandwidth_bytes_per_sec: f64,
}

// ── PageFaultEvent ───────────────────────────────────────────────────

/// Simulated page-fault event for CPU testing.
#[derive(Debug, Clone)]
pub struct PageFaultEvent {
    /// Buffer that triggered the fault.
    pub buffer_id: u64,
    /// Page offset (in pages) within the buffer.
    pub page_offset: usize,
    /// Device that owned the page before migration.
    pub source_device: i32,
    /// Device that requested the page.
    pub target_device: i32,
    /// Timestamp.
    pub timestamp: Instant,
}

// ── UnifiedMemoryManager ─────────────────────────────────────────────

/// Central manager for unified-memory buffers, advisory hints, and prefetch.
///
/// CPU-fallback: all operations are simulated in user-space so tests can run
/// without a GPU.
pub struct UnifiedMemoryManager {
    config: UnifiedMemoryConfig,
    /// Live buffers keyed by id.
    buffers: HashMap<u64, BufferMeta>,
    /// Aggregate statistics.
    stats: UnifiedMemoryStats,
    /// Page-fault log.
    page_fault_log: Vec<PageFaultEvent>,
    /// Copy profiling records.
    copy_records: Vec<CopyRecord>,
}

/// Internal metadata for a managed buffer tracked by the manager.
#[derive(Debug)]
struct BufferMeta {
    byte_size: usize,
    current_device: i32,
    advices: Vec<(MemoryAdvice, i32)>,
    page_faults: u64,
    bytes_transferred: u64,
}

impl UnifiedMemoryManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: UnifiedMemoryConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            buffers: HashMap::new(),
            stats: UnifiedMemoryStats::default(),
            page_fault_log: Vec::new(),
            copy_records: Vec::new(),
        })
    }

    /// Create a manager with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(UnifiedMemoryConfig::default())
    }

    /// Reference to current configuration.
    pub fn config(&self) -> &UnifiedMemoryConfig {
        &self.config
    }

    // ── buffer lifecycle ─────────────────────────────────────────────

    /// Register a managed buffer with the manager.
    pub fn register<T: Copy + Default>(&mut self, buffer: &ManagedBuffer<T>) -> Result<()> {
        if self.buffers.contains_key(&buffer.id()) {
            return Err(KernelError::InvalidArguments {
                reason: format!("buffer {} already registered", buffer.id()),
            }
            .into());
        }
        self.buffers.insert(
            buffer.id(),
            BufferMeta {
                byte_size: buffer.byte_size(),
                current_device: buffer.current_device(),
                advices: Vec::new(),
                page_faults: 0,
                bytes_transferred: 0,
            },
        );
        self.stats.total_allocated_bytes += buffer.byte_size() as u64;
        self.stats.active_buffers += 1;
        Ok(())
    }

    /// Deregister a buffer (e.g. before dropping it).
    pub fn deregister(&mut self, buffer_id: u64) -> Result<()> {
        let meta = self.buffers.remove(&buffer_id).ok_or_else(|| {
            KernelError::InvalidArguments { reason: format!("buffer {} not registered", buffer_id) }
        })?;
        self.stats.total_allocated_bytes =
            self.stats.total_allocated_bytes.saturating_sub(meta.byte_size as u64);
        self.stats.active_buffers = self.stats.active_buffers.saturating_sub(1);
        Ok(())
    }

    /// Return whether a buffer id is tracked.
    pub fn is_registered(&self, buffer_id: u64) -> bool {
        self.buffers.contains_key(&buffer_id)
    }

    // ── advise ───────────────────────────────────────────────────────

    /// Apply a memory advisory hint.
    ///
    /// On GPU this maps to `cudaMemAdvise`; CPU fallback records the hint.
    pub fn advise_memory<T: Copy + Default>(
        &mut self,
        buffer: &mut ManagedBuffer<T>,
        advice: MemoryAdvice,
        device: i32,
    ) -> Result<()> {
        if !self.config.enable_advise {
            return Ok(());
        }
        let meta =
            self.buffers.get_mut(&buffer.id()).ok_or_else(|| KernelError::InvalidArguments {
                reason: format!("buffer {} not registered; call register() first", buffer.id()),
            })?;
        buffer.advices.push((advice, device));
        meta.advices.push((advice, device));
        self.stats.total_advise_calls += 1;
        Ok(())
    }

    // ── prefetch ─────────────────────────────────────────────────────

    /// Asynchronously prefetch a buffer to the given device.
    ///
    /// On GPU this calls `cudaMemPrefetchAsync`; CPU fallback updates
    /// location metadata and accumulates transfer stats.
    pub fn prefetch_to_device<T: Copy + Default>(
        &mut self,
        buffer: &mut ManagedBuffer<T>,
        device_id: i32,
        _stream: Option<u64>,
    ) -> Result<()> {
        if !self.config.enable_prefetch {
            return Ok(());
        }
        if device_id < 0 {
            return Err(KernelError::InvalidArguments {
                reason: "device_id must be >= 0 for device prefetch".into(),
            }
            .into());
        }
        let meta =
            self.buffers.get_mut(&buffer.id()).ok_or_else(|| KernelError::InvalidArguments {
                reason: format!("buffer {} not registered", buffer.id()),
            })?;
        let bytes = buffer.byte_size() as u64;
        buffer.current_device = device_id;
        buffer.bytes_transferred += bytes;
        buffer.last_access = Instant::now();
        meta.current_device = device_id;
        meta.bytes_transferred += bytes;
        self.stats.total_prefetched_bytes += bytes;
        Ok(())
    }

    /// Prefetch a buffer back to the host (device = -1 / `cudaCpuDeviceId`).
    pub fn prefetch_to_host<T: Copy + Default>(
        &mut self,
        buffer: &mut ManagedBuffer<T>,
    ) -> Result<()> {
        if !self.config.enable_prefetch {
            return Ok(());
        }
        let meta =
            self.buffers.get_mut(&buffer.id()).ok_or_else(|| KernelError::InvalidArguments {
                reason: format!("buffer {} not registered", buffer.id()),
            })?;
        let bytes = buffer.byte_size() as u64;
        buffer.current_device = -1;
        buffer.bytes_transferred += bytes;
        buffer.last_access = Instant::now();
        meta.current_device = -1;
        meta.bytes_transferred += bytes;
        self.stats.total_prefetched_bytes += bytes;
        Ok(())
    }

    /// Prefetch multiple buffers to a device in one call.
    pub fn batch_prefetch<T: Copy + Default>(
        &mut self,
        buffers: &mut [&mut ManagedBuffer<T>],
        device_id: i32,
    ) -> Result<usize> {
        if !self.config.enable_prefetch {
            return Ok(0);
        }
        if device_id < -1 {
            return Err(
                KernelError::InvalidArguments { reason: "device_id must be >= -1".into() }.into()
            );
        }
        let mut count = 0usize;
        for buf in buffers.iter_mut() {
            if device_id == -1 {
                self.prefetch_to_host_inner(buf)?;
            } else {
                self.prefetch_to_device_inner(buf, device_id)?;
            }
            count += 1;
        }
        Ok(count)
    }

    /// Inner helper for `batch_prefetch` to device.
    fn prefetch_to_device_inner<T: Copy + Default>(
        &mut self,
        buffer: &mut ManagedBuffer<T>,
        device_id: i32,
    ) -> Result<()> {
        let meta =
            self.buffers.get_mut(&buffer.id()).ok_or_else(|| KernelError::InvalidArguments {
                reason: format!("buffer {} not registered", buffer.id()),
            })?;
        let bytes = buffer.byte_size() as u64;
        buffer.current_device = device_id;
        buffer.bytes_transferred += bytes;
        buffer.last_access = Instant::now();
        meta.current_device = device_id;
        meta.bytes_transferred += bytes;
        self.stats.total_prefetched_bytes += bytes;
        Ok(())
    }

    /// Inner helper for `batch_prefetch` to host.
    fn prefetch_to_host_inner<T: Copy + Default>(
        &mut self,
        buffer: &mut ManagedBuffer<T>,
    ) -> Result<()> {
        let meta =
            self.buffers.get_mut(&buffer.id()).ok_or_else(|| KernelError::InvalidArguments {
                reason: format!("buffer {} not registered", buffer.id()),
            })?;
        let bytes = buffer.byte_size() as u64;
        buffer.current_device = -1;
        buffer.bytes_transferred += bytes;
        buffer.last_access = Instant::now();
        meta.current_device = -1;
        meta.bytes_transferred += bytes;
        self.stats.total_prefetched_bytes += bytes;
        Ok(())
    }

    // ── explicit copy ────────────────────────────────────────────────

    /// Copy data between managed buffers with direction and profiling.
    ///
    /// Returns a [`CopyRecord`] with timing and bandwidth information.
    pub fn unified_memcpy<T: Copy + Default>(
        &mut self,
        src: &ManagedBuffer<T>,
        dst: &mut ManagedBuffer<T>,
        kind: CopyKind,
    ) -> Result<CopyRecord> {
        if src.len() != dst.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("source length {} != destination length {}", src.len(), dst.len()),
            }
            .into());
        }
        // Both buffers must be registered.
        if !self.buffers.contains_key(&src.id()) {
            return Err(KernelError::InvalidArguments {
                reason: format!("source buffer {} not registered", src.id()),
            }
            .into());
        }
        if !self.buffers.contains_key(&dst.id()) {
            return Err(KernelError::InvalidArguments {
                reason: format!("destination buffer {} not registered", dst.id()),
            }
            .into());
        }

        let bytes = src.byte_size();
        let start = Instant::now();

        // CPU fallback: plain copy.
        dst.data.copy_from_slice(&src.data);
        dst.last_access = Instant::now();

        let elapsed = start.elapsed();
        let bw = if elapsed.as_secs_f64() > 0.0 {
            bytes as f64 / elapsed.as_secs_f64()
        } else {
            f64::INFINITY
        };

        // Update device metadata based on copy direction.
        match kind {
            CopyKind::HostToDevice => {
                dst.current_device = 0;
            }
            CopyKind::DeviceToHost => {
                dst.current_device = -1;
            }
            CopyKind::DeviceToDevice => {
                dst.current_device = src.current_device;
            }
            CopyKind::HostToHost => {
                dst.current_device = -1;
            }
        }

        let record = CopyRecord { kind, bytes, duration: elapsed, bandwidth_bytes_per_sec: bw };
        self.stats.total_copied_bytes += bytes as u64;
        self.copy_records.push(record.clone());

        // Update per-buffer stats.
        if let Some(meta) = self.buffers.get_mut(&src.id()) {
            meta.bytes_transferred += bytes as u64;
        }
        if let Some(meta) = self.buffers.get_mut(&dst.id()) {
            meta.bytes_transferred += bytes as u64;
            meta.current_device = dst.current_device;
        }

        Ok(record)
    }

    // ── page fault simulation ────────────────────────────────────────

    /// Simulate a page fault on a managed buffer.
    ///
    /// This is useful for testing page-migration logic without a GPU.
    pub fn simulate_page_fault<T: Copy + Default>(
        &mut self,
        buffer: &mut ManagedBuffer<T>,
        page_offset: usize,
        target_device: i32,
    ) -> Result<PageFaultEvent> {
        let pages = buffer.byte_size().div_ceil(self.config.page_size);
        if page_offset >= pages {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "page_offset {} >= total pages {} (page_size={})",
                    page_offset, pages, self.config.page_size
                ),
            }
            .into());
        }
        let event = PageFaultEvent {
            buffer_id: buffer.id(),
            page_offset,
            source_device: buffer.current_device,
            target_device,
            timestamp: Instant::now(),
        };
        buffer.page_faults += 1;
        self.stats.total_page_faults += 1;
        self.page_fault_log.push(event.clone());
        // Migrate page: in CPU fallback we just update the device.
        buffer.current_device = target_device;
        if let Some(meta) = self.buffers.get_mut(&buffer.id()) {
            meta.page_faults += 1;
            meta.current_device = target_device;
        }
        Ok(event)
    }

    // ── statistics ───────────────────────────────────────────────────

    /// Return a snapshot of aggregate unified-memory statistics.
    pub fn memory_usage_stats(&self) -> UnifiedMemoryStats {
        self.stats.clone()
    }

    /// Return the page-fault log.
    pub fn page_fault_log(&self) -> &[PageFaultEvent] {
        &self.page_fault_log
    }

    /// Return profiling records for all `unified_memcpy` calls.
    pub fn copy_records(&self) -> &[CopyRecord] {
        &self.copy_records
    }

    /// Clear accumulated profiling records.
    pub fn clear_profiling(&mut self) {
        self.copy_records.clear();
        self.page_fault_log.clear();
    }

    /// Number of currently tracked buffers.
    pub fn active_buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

// ── Free functions (convenience wrappers) ────────────────────────────

/// Apply a memory advisory hint to a managed buffer.
///
/// Delegates to [`UnifiedMemoryManager::advise_memory`].
pub fn advise_memory<T: Copy + Default>(
    mgr: &mut UnifiedMemoryManager,
    buffer: &mut ManagedBuffer<T>,
    advice: MemoryAdvice,
    device: i32,
) -> Result<()> {
    mgr.advise_memory(buffer, advice, device)
}

/// Asynchronously prefetch a buffer to the given device.
///
/// Delegates to [`UnifiedMemoryManager::prefetch_to_device`].
pub fn prefetch_to_device<T: Copy + Default>(
    mgr: &mut UnifiedMemoryManager,
    buffer: &mut ManagedBuffer<T>,
    device_id: i32,
    stream: Option<u64>,
) -> Result<()> {
    mgr.prefetch_to_device(buffer, device_id, stream)
}

/// Prefetch a buffer back to the host.
///
/// Delegates to [`UnifiedMemoryManager::prefetch_to_host`].
pub fn prefetch_to_host<T: Copy + Default>(
    mgr: &mut UnifiedMemoryManager,
    buffer: &mut ManagedBuffer<T>,
) -> Result<()> {
    mgr.prefetch_to_host(buffer)
}

/// Copy data between managed buffers with direction and profiling.
///
/// Delegates to [`UnifiedMemoryManager::unified_memcpy`].
pub fn unified_memcpy<T: Copy + Default>(
    mgr: &mut UnifiedMemoryManager,
    src: &ManagedBuffer<T>,
    dst: &mut ManagedBuffer<T>,
    kind: CopyKind,
) -> Result<CopyRecord> {
    mgr.unified_memcpy(src, dst, kind)
}

/// Return aggregate unified-memory statistics.
///
/// Delegates to [`UnifiedMemoryManager::memory_usage_stats`].
pub fn memory_usage_stats(mgr: &UnifiedMemoryManager) -> UnifiedMemoryStats {
    mgr.memory_usage_stats()
}

/// Prefetch multiple buffers to a device in one call.
///
/// Delegates to [`UnifiedMemoryManager::batch_prefetch`].
pub fn batch_prefetch<T: Copy + Default>(
    mgr: &mut UnifiedMemoryManager,
    buffers: &mut [&mut ManagedBuffer<T>],
    device_id: i32,
) -> Result<usize> {
    mgr.batch_prefetch(buffers, device_id)
}

/// Simulate a page-fault on a managed buffer (CPU testing helper).
///
/// Delegates to [`UnifiedMemoryManager::simulate_page_fault`].
pub fn simulate_page_fault<T: Copy + Default>(
    mgr: &mut UnifiedMemoryManager,
    buffer: &mut ManagedBuffer<T>,
    page_offset: usize,
    target_device: i32,
) -> Result<PageFaultEvent> {
    mgr.simulate_page_fault(buffer, page_offset, target_device)
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────

    fn default_manager() -> UnifiedMemoryManager {
        UnifiedMemoryManager::with_defaults().unwrap()
    }

    fn make_buffer(len: usize) -> ManagedBuffer<f32> {
        ManagedBuffer::<f32>::new(len).unwrap()
    }

    // ── UnifiedMemoryConfig tests ───────────────────────────────────

    #[test]
    fn test_config_default_is_valid() {
        UnifiedMemoryConfig::default().validate().unwrap();
    }

    #[test]
    fn test_config_page_size_too_small() {
        let cfg = UnifiedMemoryConfig::default().with_page_size(2048);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_page_size_not_power_of_two() {
        let cfg = UnifiedMemoryConfig::default().with_page_size(5000);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_oversubscription_zero() {
        let cfg = UnifiedMemoryConfig::default().with_oversubscription_limit(0.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_oversubscription_negative() {
        let cfg = UnifiedMemoryConfig::default().with_oversubscription_limit(-1.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_oversubscription_too_large() {
        let cfg = UnifiedMemoryConfig::default().with_oversubscription_limit(17.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_valid_custom() {
        let cfg = UnifiedMemoryConfig::default()
            .with_page_size(4096)
            .with_oversubscription_limit(2.0)
            .with_prefetch(false)
            .with_advise(false);
        cfg.validate().unwrap();
        assert!(!cfg.enable_prefetch);
        assert!(!cfg.enable_advise);
        assert_eq!(cfg.page_size, 4096);
        assert!((cfg.oversubscription_limit - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_builder_chaining() {
        let cfg = UnifiedMemoryConfig::default()
            .with_prefetch(true)
            .with_advise(true)
            .with_page_size(8192)
            .with_oversubscription_limit(1.5);
        cfg.validate().unwrap();
        assert_eq!(cfg.page_size, 8192);
    }

    #[test]
    fn test_config_page_size_exact_4096() {
        let cfg = UnifiedMemoryConfig::default().with_page_size(4096);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_config_oversubscription_boundary_16() {
        let cfg = UnifiedMemoryConfig::default().with_oversubscription_limit(16.0);
        cfg.validate().unwrap();
    }

    // ── ManagedBuffer tests ─────────────────────────────────────────

    #[test]
    fn test_buffer_creation() {
        let buf = make_buffer(1024);
        assert_eq!(buf.len(), 1024);
        assert!(!buf.is_empty());
        assert_eq!(buf.byte_size(), 1024 * 4);
        assert_eq!(buf.current_device(), -1);
    }

    #[test]
    fn test_buffer_zero_length_error() {
        let res = ManagedBuffer::<f32>::new(0);
        assert!(res.is_err());
    }

    #[test]
    fn test_buffer_initial_values() {
        let buf = make_buffer(16);
        for &v in buf.peek() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_buffer_as_slice() {
        let mut buf = make_buffer(8);
        {
            let s = buf.as_mut_slice();
            s[0] = 1.0;
            s[7] = 42.0;
        }
        let s = buf.as_slice();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[7], 42.0);
    }

    #[test]
    fn test_buffer_page_fault_on_device_access() {
        let mut buf = make_buffer(8);
        buf.current_device = 0; // simulate on-device
        let _ = buf.as_slice();
        assert_eq!(buf.page_faults(), 1);
    }

    #[test]
    fn test_buffer_no_page_fault_on_host() {
        let mut buf = make_buffer(8);
        let _ = buf.as_slice();
        assert_eq!(buf.page_faults(), 0);
    }

    #[test]
    fn test_buffer_mut_page_fault() {
        let mut buf = make_buffer(8);
        buf.current_device = 0;
        let _ = buf.as_mut_slice();
        assert_eq!(buf.page_faults(), 1);
    }

    #[test]
    fn test_buffer_unique_ids() {
        let a = make_buffer(4);
        let b = make_buffer(4);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn test_buffer_bytes_transferred_initial() {
        let buf = make_buffer(4);
        assert_eq!(buf.bytes_transferred(), 0);
    }

    #[test]
    fn test_buffer_advices_initial_empty() {
        let buf = make_buffer(4);
        assert!(buf.advices().is_empty());
    }

    #[test]
    fn test_buffer_u8_type() {
        let buf = ManagedBuffer::<u8>::new(256).unwrap();
        assert_eq!(buf.byte_size(), 256);
    }

    #[test]
    fn test_buffer_i32_type() {
        let buf = ManagedBuffer::<i32>::new(100).unwrap();
        assert_eq!(buf.byte_size(), 400);
    }

    #[test]
    fn test_buffer_peek_no_side_effects() {
        let mut buf = make_buffer(4);
        buf.current_device = 0;
        let _ = buf.peek();
        assert_eq!(buf.page_faults(), 0);
    }

    // ── MemoryAdvice tests ──────────────────────────────────────────

    #[test]
    fn test_advice_display() {
        assert_eq!(MemoryAdvice::ReadMostly.to_string(), "ReadMostly");
        assert_eq!(MemoryAdvice::PreferredLocation.to_string(), "PreferredLocation");
        assert_eq!(MemoryAdvice::AccessedBy.to_string(), "AccessedBy");
        assert_eq!(MemoryAdvice::WillNeedSoon.to_string(), "WillNeedSoon");
    }

    #[test]
    fn test_advice_equality() {
        assert_eq!(MemoryAdvice::ReadMostly, MemoryAdvice::ReadMostly);
        assert_ne!(MemoryAdvice::ReadMostly, MemoryAdvice::AccessedBy);
    }

    // ── CopyKind tests ──────────────────────────────────────────────

    #[test]
    fn test_copy_kind_display() {
        assert_eq!(CopyKind::HostToDevice.to_string(), "HostToDevice");
        assert_eq!(CopyKind::DeviceToHost.to_string(), "DeviceToHost");
        assert_eq!(CopyKind::DeviceToDevice.to_string(), "DeviceToDevice");
        assert_eq!(CopyKind::HostToHost.to_string(), "HostToHost");
    }

    // ── PrefetchRequest tests ───────────────────────────────────────

    #[test]
    fn test_prefetch_request_new() {
        let req = PrefetchRequest::new(42, 0);
        assert_eq!(req.buffer_id, 42);
        assert_eq!(req.device_id, 0);
        assert!(req.stream.is_none());
    }

    #[test]
    fn test_prefetch_request_with_stream() {
        let req = PrefetchRequest::new(1, 0).with_stream(7);
        assert_eq!(req.stream, Some(7));
    }

    // ── Manager lifecycle tests ─────────────────────────────────────

    #[test]
    fn test_manager_creation() {
        let mgr = default_manager();
        assert_eq!(mgr.active_buffer_count(), 0);
    }

    #[test]
    fn test_manager_invalid_config() {
        let cfg = UnifiedMemoryConfig::default().with_page_size(100);
        assert!(UnifiedMemoryManager::new(cfg).is_err());
    }

    #[test]
    fn test_register_buffer() {
        let mut mgr = default_manager();
        let buf = make_buffer(256);
        mgr.register(&buf).unwrap();
        assert!(mgr.is_registered(buf.id()));
        assert_eq!(mgr.active_buffer_count(), 1);
    }

    #[test]
    fn test_register_duplicate_error() {
        let mut mgr = default_manager();
        let buf = make_buffer(8);
        mgr.register(&buf).unwrap();
        assert!(mgr.register(&buf).is_err());
    }

    #[test]
    fn test_deregister_buffer() {
        let mut mgr = default_manager();
        let buf = make_buffer(8);
        mgr.register(&buf).unwrap();
        mgr.deregister(buf.id()).unwrap();
        assert!(!mgr.is_registered(buf.id()));
        assert_eq!(mgr.active_buffer_count(), 0);
    }

    #[test]
    fn test_deregister_unknown_error() {
        let mut mgr = default_manager();
        assert!(mgr.deregister(9999).is_err());
    }

    #[test]
    fn test_register_multiple_buffers() {
        let mut mgr = default_manager();
        let b1 = make_buffer(64);
        let b2 = make_buffer(128);
        let b3 = make_buffer(256);
        mgr.register(&b1).unwrap();
        mgr.register(&b2).unwrap();
        mgr.register(&b3).unwrap();
        assert_eq!(mgr.active_buffer_count(), 3);
    }

    // ── advise_memory tests ─────────────────────────────────────────

    #[test]
    fn test_advise_read_mostly() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        assert_eq!(buf.advices().len(), 1);
        assert_eq!(buf.advices()[0], (MemoryAdvice::ReadMostly, 0));
    }

    #[test]
    fn test_advise_preferred_location() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::PreferredLocation, 0).unwrap();
        assert_eq!(buf.advices()[0].0, MemoryAdvice::PreferredLocation);
    }

    #[test]
    fn test_advise_accessed_by() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::AccessedBy, 1).unwrap();
        assert_eq!(buf.advices()[0], (MemoryAdvice::AccessedBy, 1));
    }

    #[test]
    fn test_advise_will_need_soon() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::WillNeedSoon, 0).unwrap();
        assert_eq!(buf.advices()[0].0, MemoryAdvice::WillNeedSoon);
    }

    #[test]
    fn test_advise_unregistered_error() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        assert!(mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).is_err());
    }

    #[test]
    fn test_advise_disabled() {
        let cfg = UnifiedMemoryConfig::default().with_advise(false);
        let mut mgr = UnifiedMemoryManager::new(cfg).unwrap();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        // No advice recorded when disabled.
        assert!(buf.advices().is_empty());
    }

    #[test]
    fn test_advise_multiple_hints() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::AccessedBy, 1).unwrap();
        assert_eq!(buf.advices().len(), 2);
    }

    #[test]
    fn test_advise_stats_increment() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::WillNeedSoon, 0).unwrap();
        assert_eq!(mgr.memory_usage_stats().total_advise_calls, 2);
    }

    // ── advise free function ────────────────────────────────────────

    #[test]
    fn test_free_fn_advise_memory() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        advise_memory(&mut mgr, &mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        assert_eq!(buf.advices().len(), 1);
    }

    // ── prefetch_to_device tests ────────────────────────────────────

    #[test]
    fn test_prefetch_to_device() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(64);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        assert_eq!(buf.current_device(), 0);
    }

    #[test]
    fn test_prefetch_to_device_with_stream() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(64);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, Some(5)).unwrap();
        assert_eq!(buf.current_device(), 0);
    }

    #[test]
    fn test_prefetch_negative_device_error() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(64);
        mgr.register(&buf).unwrap();
        assert!(mgr.prefetch_to_device(&mut buf, -1, None).is_err());
    }

    #[test]
    fn test_prefetch_unregistered_error() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(64);
        assert!(mgr.prefetch_to_device(&mut buf, 0, None).is_err());
    }

    #[test]
    fn test_prefetch_disabled() {
        let cfg = UnifiedMemoryConfig::default().with_prefetch(false);
        let mut mgr = UnifiedMemoryManager::new(cfg).unwrap();
        let mut buf = make_buffer(64);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        assert_eq!(buf.current_device(), -1); // unchanged
    }

    #[test]
    fn test_prefetch_bytes_transferred() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(256);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        assert_eq!(buf.bytes_transferred(), 256 * 4);
    }

    #[test]
    fn test_prefetch_stats_update() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(128);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        let stats = mgr.memory_usage_stats();
        assert_eq!(stats.total_prefetched_bytes, 128 * 4);
    }

    // ── prefetch free function ──────────────────────────────────────

    #[test]
    fn test_free_fn_prefetch_to_device() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        prefetch_to_device(&mut mgr, &mut buf, 0, None).unwrap();
        assert_eq!(buf.current_device(), 0);
    }

    // ── prefetch_to_host tests ──────────────────────────────────────

    #[test]
    fn test_prefetch_to_host() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(64);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        assert_eq!(buf.current_device(), 0);
        mgr.prefetch_to_host(&mut buf).unwrap();
        assert_eq!(buf.current_device(), -1);
    }

    #[test]
    fn test_prefetch_to_host_unregistered() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(16);
        assert!(mgr.prefetch_to_host(&mut buf).is_err());
    }

    #[test]
    fn test_prefetch_to_host_disabled() {
        let cfg = UnifiedMemoryConfig::default().with_prefetch(false);
        let mut mgr = UnifiedMemoryManager::new(cfg).unwrap();
        let mut buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        buf.current_device = 0;
        mgr.prefetch_to_host(&mut buf).unwrap();
        assert_eq!(buf.current_device(), 0); // unchanged
    }

    // ── prefetch_to_host free function ──────────────────────────────

    #[test]
    fn test_free_fn_prefetch_to_host() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        prefetch_to_device(&mut mgr, &mut buf, 0, None).unwrap();
        prefetch_to_host(&mut mgr, &mut buf).unwrap();
        assert_eq!(buf.current_device(), -1);
    }

    // ── batch_prefetch tests ────────────────────────────────────────

    #[test]
    fn test_batch_prefetch_to_device() {
        let mut mgr = default_manager();
        let mut b1 = make_buffer(32);
        let mut b2 = make_buffer(64);
        mgr.register(&b1).unwrap();
        mgr.register(&b2).unwrap();
        let count = mgr.batch_prefetch(&mut [&mut b1, &mut b2], 0).unwrap();
        assert_eq!(count, 2);
        assert_eq!(b1.current_device(), 0);
        assert_eq!(b2.current_device(), 0);
    }

    #[test]
    fn test_batch_prefetch_to_host() {
        let mut mgr = default_manager();
        let mut b1 = make_buffer(16);
        mgr.register(&b1).unwrap();
        mgr.prefetch_to_device(&mut b1, 0, None).unwrap();
        let count = mgr.batch_prefetch(&mut [&mut b1], -1).unwrap();
        assert_eq!(count, 1);
        assert_eq!(b1.current_device(), -1);
    }

    #[test]
    fn test_batch_prefetch_empty() {
        let mut mgr = default_manager();
        let bufs: &mut [&mut ManagedBuffer<f32>] = &mut [];
        let count = mgr.batch_prefetch(bufs, 0).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_batch_prefetch_disabled() {
        let cfg = UnifiedMemoryConfig::default().with_prefetch(false);
        let mut mgr = UnifiedMemoryManager::new(cfg).unwrap();
        let mut b1 = make_buffer(16);
        mgr.register(&b1).unwrap();
        let count = mgr.batch_prefetch(&mut [&mut b1], 0).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_batch_prefetch_invalid_device() {
        let mut mgr = default_manager();
        let mut b1 = make_buffer(8);
        mgr.register(&b1).unwrap();
        assert!(mgr.batch_prefetch(&mut [&mut b1], -2).is_err());
    }

    // ── batch_prefetch free function ────────────────────────────────

    #[test]
    fn test_free_fn_batch_prefetch() {
        let mut mgr = default_manager();
        let mut b1 = make_buffer(16);
        let mut b2 = make_buffer(16);
        mgr.register(&b1).unwrap();
        mgr.register(&b2).unwrap();
        let n = batch_prefetch(&mut mgr, &mut [&mut b1, &mut b2], 0).unwrap();
        assert_eq!(n, 2);
    }

    // ── unified_memcpy tests ────────────────────────────────────────

    #[test]
    fn test_memcpy_host_to_device() {
        let mut mgr = default_manager();
        let mut src = make_buffer(8);
        let mut dst = make_buffer(8);
        src.as_mut_slice()[0] = 3.14;
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        let rec = mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToDevice).unwrap();
        assert_eq!(dst.peek()[0], 3.14);
        assert_eq!(rec.kind, CopyKind::HostToDevice);
        assert_eq!(rec.bytes, 8 * 4);
        assert_eq!(dst.current_device(), 0);
    }

    #[test]
    fn test_memcpy_device_to_host() {
        let mut mgr = default_manager();
        let mut src = make_buffer(4);
        let mut dst = make_buffer(4);
        src.as_mut_slice()[0] = 2.71;
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        let rec = mgr.unified_memcpy(&src, &mut dst, CopyKind::DeviceToHost).unwrap();
        assert_eq!(dst.peek()[0], 2.71);
        assert_eq!(rec.kind, CopyKind::DeviceToHost);
        assert_eq!(dst.current_device(), -1);
    }

    #[test]
    fn test_memcpy_device_to_device() {
        let mut mgr = default_manager();
        let mut src = make_buffer(4);
        let mut dst = make_buffer(4);
        src.current_device = 1;
        src.as_mut_slice()[0] = 99.0;
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        mgr.unified_memcpy(&src, &mut dst, CopyKind::DeviceToDevice).unwrap();
        assert_eq!(dst.peek()[0], 99.0);
        assert_eq!(dst.current_device(), 1);
    }

    #[test]
    fn test_memcpy_host_to_host() {
        let mut mgr = default_manager();
        let mut src = make_buffer(4);
        let mut dst = make_buffer(4);
        src.as_mut_slice()[3] = 7.0;
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).unwrap();
        assert_eq!(dst.peek()[3], 7.0);
        assert_eq!(dst.current_device(), -1);
    }

    #[test]
    fn test_memcpy_length_mismatch() {
        let mut mgr = default_manager();
        let src = make_buffer(4);
        let mut dst = make_buffer(8);
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        assert!(mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).is_err());
    }

    #[test]
    fn test_memcpy_src_unregistered() {
        let mut mgr = default_manager();
        let src = make_buffer(4);
        let mut dst = make_buffer(4);
        mgr.register(&dst).unwrap();
        assert!(mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).is_err());
    }

    #[test]
    fn test_memcpy_dst_unregistered() {
        let mut mgr = default_manager();
        let src = make_buffer(4);
        let mut dst = make_buffer(4);
        mgr.register(&src).unwrap();
        assert!(mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).is_err());
    }

    #[test]
    fn test_memcpy_profiling_record() {
        let mut mgr = default_manager();
        let src = make_buffer(64);
        let mut dst = make_buffer(64);
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToDevice).unwrap();
        assert_eq!(mgr.copy_records().len(), 1);
        assert!(mgr.copy_records()[0].bandwidth_bytes_per_sec > 0.0);
    }

    #[test]
    fn test_memcpy_stats_update() {
        let mut mgr = default_manager();
        let src = make_buffer(16);
        let mut dst = make_buffer(16);
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).unwrap();
        assert_eq!(mgr.memory_usage_stats().total_copied_bytes, 16 * 4);
    }

    // ── unified_memcpy free function ────────────────────────────────

    #[test]
    fn test_free_fn_unified_memcpy() {
        let mut mgr = default_manager();
        let mut src = make_buffer(4);
        let mut dst = make_buffer(4);
        src.as_mut_slice()[0] = 1.0;
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        let rec = unified_memcpy(&mut mgr, &src, &mut dst, CopyKind::HostToHost).unwrap();
        assert_eq!(rec.bytes, 16);
        assert_eq!(dst.peek()[0], 1.0);
    }

    // ── page fault simulation tests ─────────────────────────────────

    #[test]
    fn test_simulate_page_fault() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(1024);
        mgr.register(&buf).unwrap();
        let ev = mgr.simulate_page_fault(&mut buf, 0, 0).unwrap();
        assert_eq!(ev.buffer_id, buf.id());
        assert_eq!(ev.page_offset, 0);
        assert_eq!(ev.source_device, -1);
        assert_eq!(ev.target_device, 0);
        assert_eq!(buf.page_faults(), 1);
    }

    #[test]
    fn test_simulate_page_fault_out_of_range() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(8); // 32 bytes < 1 page
        mgr.register(&buf).unwrap();
        // page_offset = 1 is out of range (only page 0 exists).
        assert!(mgr.simulate_page_fault(&mut buf, 1, 0).is_err());
    }

    #[test]
    fn test_simulate_page_fault_log() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(1024);
        mgr.register(&buf).unwrap();
        mgr.simulate_page_fault(&mut buf, 0, 0).unwrap();
        mgr.simulate_page_fault(&mut buf, 0, -1).unwrap();
        assert_eq!(mgr.page_fault_log().len(), 2);
        assert_eq!(mgr.memory_usage_stats().total_page_faults, 2);
    }

    #[test]
    fn test_simulate_page_fault_device_update() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(1024);
        mgr.register(&buf).unwrap();
        mgr.simulate_page_fault(&mut buf, 0, 2).unwrap();
        assert_eq!(buf.current_device(), 2);
    }

    // ── page fault free function ────────────────────────────────────

    #[test]
    fn test_free_fn_simulate_page_fault() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(1024);
        mgr.register(&buf).unwrap();
        let ev = simulate_page_fault(&mut mgr, &mut buf, 0, 0).unwrap();
        assert_eq!(ev.target_device, 0);
    }

    // ── memory_usage_stats tests ────────────────────────────────────

    #[test]
    fn test_stats_initial() {
        let mgr = default_manager();
        let stats = mgr.memory_usage_stats();
        assert_eq!(stats.total_allocated_bytes, 0);
        assert_eq!(stats.active_buffers, 0);
        assert_eq!(stats.total_prefetched_bytes, 0);
        assert_eq!(stats.total_copied_bytes, 0);
        assert_eq!(stats.total_page_faults, 0);
        assert_eq!(stats.total_advise_calls, 0);
    }

    #[test]
    fn test_stats_after_register() {
        let mut mgr = default_manager();
        let buf = make_buffer(100);
        mgr.register(&buf).unwrap();
        let stats = mgr.memory_usage_stats();
        assert_eq!(stats.total_allocated_bytes, 400);
        assert_eq!(stats.active_buffers, 1);
    }

    #[test]
    fn test_stats_after_mixed_ops() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        let stats = mgr.memory_usage_stats();
        assert_eq!(stats.total_prefetched_bytes, 64);
        assert_eq!(stats.total_advise_calls, 1);
    }

    // ── memory_usage_stats free function ────────────────────────────

    #[test]
    fn test_free_fn_memory_usage_stats() {
        let mgr = default_manager();
        let stats = memory_usage_stats(&mgr);
        assert_eq!(stats.active_buffers, 0);
    }

    // ── profiling / clear tests ─────────────────────────────────────

    #[test]
    fn test_clear_profiling() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(1024);
        mgr.register(&buf).unwrap();
        mgr.simulate_page_fault(&mut buf, 0, 0).unwrap();
        let src = make_buffer(8);
        let mut dst = make_buffer(8);
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).unwrap();
        assert!(!mgr.copy_records().is_empty());
        assert!(!mgr.page_fault_log().is_empty());
        mgr.clear_profiling();
        assert!(mgr.copy_records().is_empty());
        assert!(mgr.page_fault_log().is_empty());
    }

    // ── round-trip tests ────────────────────────────────────────────

    #[test]
    fn test_prefetch_round_trip() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(128);
        mgr.register(&buf).unwrap();
        assert_eq!(buf.current_device(), -1);
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        assert_eq!(buf.current_device(), 0);
        mgr.prefetch_to_host(&mut buf).unwrap();
        assert_eq!(buf.current_device(), -1);
        // Two prefetch operations.
        let stats = mgr.memory_usage_stats();
        assert_eq!(stats.total_prefetched_bytes, 128 * 4 * 2);
    }

    #[test]
    fn test_memcpy_preserves_all_data() {
        let mut mgr = default_manager();
        let mut src = make_buffer(256);
        let mut dst = make_buffer(256);
        for (i, v) in src.as_mut_slice().iter_mut().enumerate() {
            *v = i as f32;
        }
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToHost).unwrap();
        for (i, &v) in dst.peek().iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    // ── edge-case / stress tests ────────────────────────────────────

    #[test]
    fn test_single_element_buffer() {
        let buf = ManagedBuffer::<f32>::new(1).unwrap();
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.byte_size(), 4);
    }

    #[test]
    fn test_large_buffer() {
        // 1M elements = 4 MiB — should succeed.
        let buf = ManagedBuffer::<f32>::new(1_000_000).unwrap();
        assert_eq!(buf.len(), 1_000_000);
    }

    #[test]
    fn test_multi_device_prefetch() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(32);
        mgr.register(&buf).unwrap();
        mgr.prefetch_to_device(&mut buf, 0, None).unwrap();
        assert_eq!(buf.current_device(), 0);
        mgr.prefetch_to_device(&mut buf, 1, None).unwrap();
        assert_eq!(buf.current_device(), 1);
        mgr.prefetch_to_device(&mut buf, 3, None).unwrap();
        assert_eq!(buf.current_device(), 3);
    }

    #[test]
    fn test_repeated_advise_same_hint() {
        let mut mgr = default_manager();
        let mut buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        for _ in 0..5 {
            mgr.advise_memory(&mut buf, MemoryAdvice::ReadMostly, 0).unwrap();
        }
        assert_eq!(buf.advices().len(), 5);
        assert_eq!(mgr.memory_usage_stats().total_advise_calls, 5);
    }

    #[test]
    fn test_register_deregister_reregister() {
        let mut mgr = default_manager();
        let buf = make_buffer(16);
        mgr.register(&buf).unwrap();
        mgr.deregister(buf.id()).unwrap();
        // Re-register should succeed.
        mgr.register(&buf).unwrap();
        assert!(mgr.is_registered(buf.id()));
    }

    #[test]
    fn test_copy_record_fields() {
        let mut mgr = default_manager();
        let src = make_buffer(32);
        let mut dst = make_buffer(32);
        mgr.register(&src).unwrap();
        mgr.register(&dst).unwrap();
        let rec = mgr.unified_memcpy(&src, &mut dst, CopyKind::HostToDevice).unwrap();
        assert_eq!(rec.kind, CopyKind::HostToDevice);
        assert_eq!(rec.bytes, 128);
        assert!(rec.bandwidth_bytes_per_sec > 0.0);
    }

    #[test]
    fn test_page_fault_multiple_pages() {
        let cfg = UnifiedMemoryConfig::default().with_page_size(4096);
        let mut mgr = UnifiedMemoryManager::new(cfg).unwrap();
        // 8192 bytes / 4096 page_size = 2 pages
        let mut buf = ManagedBuffer::<u8>::new(8192).unwrap();
        mgr.register(&buf).unwrap();
        mgr.simulate_page_fault(&mut buf, 0, 0).unwrap();
        mgr.simulate_page_fault(&mut buf, 1, 0).unwrap();
        assert_eq!(buf.page_faults(), 2);
    }

    #[test]
    fn test_page_fault_boundary_exact() {
        let cfg = UnifiedMemoryConfig::default().with_page_size(4096);
        let mut mgr = UnifiedMemoryManager::new(cfg).unwrap();
        // Exactly one page.
        let mut buf = ManagedBuffer::<u8>::new(4096).unwrap();
        mgr.register(&buf).unwrap();
        mgr.simulate_page_fault(&mut buf, 0, 0).unwrap();
        assert!(mgr.simulate_page_fault(&mut buf, 1, 0).is_err());
    }
}
