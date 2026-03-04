//! Efficient host↔GPU data transfer management for OpenCL workloads.
//!
//! Provides typed transfer operations (H2D, D2H, D2D), pinned host memory,
//! asynchronous transfers with event completion, staging buffers for batched
//! small writes, a double-buffering transfer scheduler, zero-copy shared
//! memory, and aggregate transfer statistics.  All operations have CPU
//! reference implementations so the module compiles and tests without an
//! actual OpenCL runtime.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Id generator
// ---------------------------------------------------------------------------

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Transfer direction
// ---------------------------------------------------------------------------

/// Direction of a buffer transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

impl fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H2D"),
            Self::DeviceToHost => write!(f, "D2H"),
            Self::DeviceToDevice => write!(f, "D2D"),
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer error
// ---------------------------------------------------------------------------

/// Errors that can occur during buffer transfers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// The source buffer is too small for the requested transfer.
    SourceTooSmall { required: usize, available: usize },
    /// The destination buffer is too small for the requested transfer.
    DestinationTooSmall { required: usize, available: usize },
    /// The transfer was not yet completed.
    NotReady,
    /// Staging buffer capacity exceeded.
    StagingFull { capacity: usize },
    /// An invalid configuration was provided.
    InvalidConfig(String),
    /// Zero-copy mapping is not available on this device.
    ZeroCopyUnavailable,
    /// The buffer has already been freed.
    BufferFreed,
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceTooSmall { required, available } => {
                write!(f, "source too small: need {required} bytes, have {available}")
            }
            Self::DestinationTooSmall { required, available } => {
                write!(f, "dest too small: need {required} bytes, have {available}")
            }
            Self::NotReady => write!(f, "async transfer not yet complete"),
            Self::StagingFull { capacity } => {
                write!(f, "staging buffer full (capacity {capacity} bytes)")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::ZeroCopyUnavailable => write!(f, "zero-copy mapping unavailable"),
            Self::BufferFreed => write!(f, "buffer already freed"),
        }
    }
}

impl std::error::Error for TransferError {}

pub type TransferResult<T> = Result<T, TransferError>;

// ---------------------------------------------------------------------------
// DeviceBuffer — simulated GPU buffer
// ---------------------------------------------------------------------------

/// Simulated GPU-side buffer (CPU reference implementation).
#[derive(Debug, Clone)]
pub struct DeviceBuffer {
    id: u64,
    data: Vec<u8>,
    freed: bool,
}

impl DeviceBuffer {
    /// Allocate a device buffer of `size` bytes (zeroed).
    pub fn allocate(size: usize) -> Self {
        Self { id: next_id(), data: vec![0u8; size], freed: false }
    }

    /// Allocate from existing data.
    pub fn from_data(data: &[u8]) -> Self {
        Self { id: next_id(), data: data.to_vec(), freed: false }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_freed(&self) -> bool {
        self.freed
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Mark the buffer as freed; subsequent operations will fail.
    pub fn free(&mut self) {
        self.freed = true;
        self.data.clear();
    }

    fn check_live(&self) -> TransferResult<()> {
        if self.freed { Err(TransferError::BufferFreed) } else { Ok(()) }
    }
}

// ---------------------------------------------------------------------------
// HostToDevice
// ---------------------------------------------------------------------------

/// Upload data from host (CPU) memory to a device buffer.
pub struct HostToDevice;

impl HostToDevice {
    /// Blocking upload of `src` into `dst`.
    /// Returns the number of bytes transferred.
    pub fn transfer(src: &[u8], dst: &mut DeviceBuffer) -> TransferResult<usize> {
        dst.check_live()?;
        if src.len() > dst.data.len() {
            return Err(TransferError::DestinationTooSmall {
                required: src.len(),
                available: dst.data.len(),
            });
        }
        dst.data[..src.len()].copy_from_slice(src);
        Ok(src.len())
    }

    /// Upload all of `src` into a newly allocated device buffer.
    pub fn upload(src: &[u8]) -> DeviceBuffer {
        DeviceBuffer::from_data(src)
    }

    /// Upload a sub-range `[offset..offset+len)` of `src`.
    pub fn transfer_range(
        src: &[u8],
        dst: &mut DeviceBuffer,
        dst_offset: usize,
        len: usize,
    ) -> TransferResult<usize> {
        dst.check_live()?;
        if len > src.len() {
            return Err(TransferError::SourceTooSmall { required: len, available: src.len() });
        }
        let end = dst_offset.checked_add(len).ok_or(TransferError::DestinationTooSmall {
            required: usize::MAX,
            available: dst.data.len(),
        })?;
        if end > dst.data.len() {
            return Err(TransferError::DestinationTooSmall {
                required: end,
                available: dst.data.len(),
            });
        }
        dst.data[dst_offset..end].copy_from_slice(&src[..len]);
        Ok(len)
    }
}

// ---------------------------------------------------------------------------
// DeviceToHost
// ---------------------------------------------------------------------------

/// Download data from a device buffer to host (CPU) memory.
pub struct DeviceToHost;

impl DeviceToHost {
    /// Blocking download of entire device buffer into a new `Vec<u8>`.
    pub fn download(src: &DeviceBuffer) -> TransferResult<Vec<u8>> {
        src.check_live()?;
        Ok(src.data.clone())
    }

    /// Download into a pre-allocated host buffer.
    /// Returns the number of bytes transferred.
    pub fn transfer(src: &DeviceBuffer, dst: &mut [u8]) -> TransferResult<usize> {
        src.check_live()?;
        if src.data.len() > dst.len() {
            return Err(TransferError::DestinationTooSmall {
                required: src.data.len(),
                available: dst.len(),
            });
        }
        let n = src.data.len();
        dst[..n].copy_from_slice(&src.data);
        Ok(n)
    }

    /// Download a sub-range `[offset..offset+len)` from the device buffer.
    pub fn download_range(
        src: &DeviceBuffer,
        offset: usize,
        len: usize,
    ) -> TransferResult<Vec<u8>> {
        src.check_live()?;
        let end = offset.checked_add(len).ok_or(TransferError::SourceTooSmall {
            required: usize::MAX,
            available: src.data.len(),
        })?;
        if end > src.data.len() {
            return Err(TransferError::SourceTooSmall { required: end, available: src.data.len() });
        }
        Ok(src.data[offset..end].to_vec())
    }
}

// ---------------------------------------------------------------------------
// DeviceToDevice
// ---------------------------------------------------------------------------

/// Copy data between device buffers (same device or peer).
pub struct DeviceToDevice;

impl DeviceToDevice {
    /// Full copy of `src` into `dst`. Both must be the same size.
    pub fn copy(src: &DeviceBuffer, dst: &mut DeviceBuffer) -> TransferResult<usize> {
        src.check_live()?;
        dst.check_live()?;
        if src.data.len() > dst.data.len() {
            return Err(TransferError::DestinationTooSmall {
                required: src.data.len(),
                available: dst.data.len(),
            });
        }
        let n = src.data.len();
        dst.data[..n].copy_from_slice(&src.data);
        Ok(n)
    }

    /// Copy a sub-range from `src` into `dst` at `dst_offset`.
    pub fn copy_range(
        src: &DeviceBuffer,
        src_offset: usize,
        dst: &mut DeviceBuffer,
        dst_offset: usize,
        len: usize,
    ) -> TransferResult<usize> {
        src.check_live()?;
        dst.check_live()?;
        let src_end = src_offset.checked_add(len).ok_or(TransferError::SourceTooSmall {
            required: usize::MAX,
            available: src.data.len(),
        })?;
        if src_end > src.data.len() {
            return Err(TransferError::SourceTooSmall {
                required: src_end,
                available: src.data.len(),
            });
        }
        let dst_end = dst_offset.checked_add(len).ok_or(TransferError::DestinationTooSmall {
            required: usize::MAX,
            available: dst.data.len(),
        })?;
        if dst_end > dst.data.len() {
            return Err(TransferError::DestinationTooSmall {
                required: dst_end,
                available: dst.data.len(),
            });
        }
        dst.data[dst_offset..dst_end].copy_from_slice(&src.data[src_offset..src_end]);
        Ok(len)
    }

    /// Clone a device buffer into a new buffer.
    pub fn duplicate(src: &DeviceBuffer) -> TransferResult<DeviceBuffer> {
        src.check_live()?;
        Ok(DeviceBuffer::from_data(&src.data))
    }
}

// ---------------------------------------------------------------------------
// PinnedBuffer — page-locked host memory
// ---------------------------------------------------------------------------

/// Simulated page-locked (pinned) host memory for faster DMA.
///
/// In a real OpenCL implementation this would use `CL_MEM_ALLOC_HOST_PTR`
/// or platform-specific pinning APIs.  The CPU reference simply wraps a
/// `Vec<u8>` and tags it as "pinned" to validate API usage.
#[derive(Debug, Clone)]
pub struct PinnedBuffer {
    id: u64,
    data: Vec<u8>,
    pinned: bool,
}

impl PinnedBuffer {
    /// Allocate a pinned buffer of `size` bytes (zeroed).
    pub fn allocate(size: usize) -> Self {
        Self { id: next_id(), data: vec![0u8; size], pinned: true }
    }

    /// Create from existing data (copies into pinned region).
    pub fn from_data(data: &[u8]) -> Self {
        Self { id: next_id(), data: data.to_vec(), pinned: true }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Unpin the buffer (simulated).
    pub fn unpin(&mut self) {
        self.pinned = false;
    }

    /// Upload the pinned buffer contents to a device buffer.
    pub fn upload_to(&self, dst: &mut DeviceBuffer) -> TransferResult<usize> {
        HostToDevice::transfer(&self.data, dst)
    }

    /// Download from a device buffer into this pinned buffer.
    pub fn download_from(&mut self, src: &DeviceBuffer) -> TransferResult<usize> {
        src.check_live()?;
        if src.data.len() > self.data.len() {
            return Err(TransferError::DestinationTooSmall {
                required: src.data.len(),
                available: self.data.len(),
            });
        }
        let n = src.data.len();
        self.data[..n].copy_from_slice(&src.data);
        Ok(n)
    }

    /// Simulated bandwidth multiplier for pinned vs unpinned.
    /// Real OpenCL pinned memory typically achieves 1.5–3× DMA throughput.
    pub fn bandwidth_multiplier() -> f64 {
        2.0
    }
}

// ---------------------------------------------------------------------------
// AsyncTransfer — non-blocking transfer with event completion
// ---------------------------------------------------------------------------

/// Completion status of an asynchronous transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncStatus {
    /// Transfer is still in flight.
    Pending,
    /// Transfer completed successfully.
    Complete,
    /// Transfer failed.
    Failed,
}

/// An asynchronous (non-blocking) data transfer with event-based completion.
///
/// In the CPU reference implementation, transfers complete instantly and the
/// event is immediately signalled.
#[derive(Debug)]
pub struct AsyncTransfer {
    id: u64,
    direction: TransferDirection,
    bytes: usize,
    status: AsyncStatus,
    start_time: Instant,
    completion_time: Option<Instant>,
}

impl AsyncTransfer {
    /// Initiate an async H2D transfer.
    pub fn host_to_device(src: &[u8], dst: &mut DeviceBuffer) -> TransferResult<Self> {
        let bytes = HostToDevice::transfer(src, dst)?;
        let now = Instant::now();
        Ok(Self {
            id: next_id(),
            direction: TransferDirection::HostToDevice,
            bytes,
            status: AsyncStatus::Complete,
            start_time: now,
            completion_time: Some(now),
        })
    }

    /// Initiate an async D2H transfer — returns the transfer handle and data.
    pub fn device_to_host(src: &DeviceBuffer) -> TransferResult<(Self, Vec<u8>)> {
        let data = DeviceToHost::download(src)?;
        let bytes = data.len();
        let now = Instant::now();
        let handle = Self {
            id: next_id(),
            direction: TransferDirection::DeviceToHost,
            bytes,
            status: AsyncStatus::Complete,
            start_time: now,
            completion_time: Some(now),
        };
        Ok((handle, data))
    }

    /// Initiate an async D2D copy.
    pub fn device_to_device(src: &DeviceBuffer, dst: &mut DeviceBuffer) -> TransferResult<Self> {
        let bytes = DeviceToDevice::copy(src, dst)?;
        let now = Instant::now();
        Ok(Self {
            id: next_id(),
            direction: TransferDirection::DeviceToDevice,
            bytes,
            status: AsyncStatus::Complete,
            start_time: now,
            completion_time: Some(now),
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn direction(&self) -> TransferDirection {
        self.direction
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn status(&self) -> AsyncStatus {
        self.status
    }

    pub fn is_complete(&self) -> bool {
        self.status == AsyncStatus::Complete
    }

    /// Block until the transfer completes (no-op in CPU reference).
    pub fn wait(&mut self) -> TransferResult<()> {
        if self.status == AsyncStatus::Failed {
            return Err(TransferError::NotReady);
        }
        self.status = AsyncStatus::Complete;
        if self.completion_time.is_none() {
            self.completion_time = Some(Instant::now());
        }
        Ok(())
    }

    /// Elapsed wall-clock time for the transfer.
    pub fn elapsed(&self) -> Duration {
        match self.completion_time {
            Some(t) => t.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }
}

// ---------------------------------------------------------------------------
// StagingBuffer — batched small transfers
// ---------------------------------------------------------------------------

/// A staging area that coalesces many small writes into one large upload.
///
/// Callers push byte slices into the staging buffer, then flush the entire
/// batch to a device buffer in a single transfer.
#[derive(Debug)]
pub struct StagingBuffer {
    data: Vec<u8>,
    capacity: usize,
    /// Offsets of each staged region (start, len).
    regions: Vec<(usize, usize)>,
    flushes: u64,
}

impl StagingBuffer {
    /// Create a staging buffer with the given byte capacity.
    pub fn new(capacity: usize) -> Self {
        Self { data: Vec::with_capacity(capacity), capacity, regions: Vec::new(), flushes: 0 }
    }

    /// Stage `bytes` for a future flush.
    pub fn push(&mut self, bytes: &[u8]) -> TransferResult<usize> {
        if self.data.len() + bytes.len() > self.capacity {
            return Err(TransferError::StagingFull { capacity: self.capacity });
        }
        let start = self.data.len();
        self.data.extend_from_slice(bytes);
        self.regions.push((start, bytes.len()));
        Ok(bytes.len())
    }

    /// Number of staged regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Total staged bytes.
    pub fn staged_bytes(&self) -> usize {
        self.data.len()
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.capacity - self.data.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total number of flushes performed.
    pub fn flush_count(&self) -> u64 {
        self.flushes
    }

    /// Flush the entire staging buffer into `dst` starting at `dst_offset`.
    /// Returns the number of bytes written and clears the staging buffer.
    pub fn flush_to(&mut self, dst: &mut DeviceBuffer, dst_offset: usize) -> TransferResult<usize> {
        if self.data.is_empty() {
            return Ok(0);
        }
        HostToDevice::transfer_range(&self.data, dst, dst_offset, self.data.len())?;
        let n = self.data.len();
        self.data.clear();
        self.regions.clear();
        self.flushes += 1;
        Ok(n)
    }

    /// Discard all staged data without transferring.
    pub fn clear(&mut self) {
        self.data.clear();
        self.regions.clear();
    }

    /// Get the staged data as a byte slice (for inspection / testing).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get region metadata.
    pub fn regions(&self) -> &[(usize, usize)] {
        &self.regions
    }
}

// ---------------------------------------------------------------------------
// TransferScheduler — double-buffering overlap
// ---------------------------------------------------------------------------

/// Slot index for double-buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    A,
    B,
}

impl Slot {
    fn flip(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

/// Configuration for the double-buffering transfer scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Size of each transfer slot in bytes.
    pub slot_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { slot_size: 1024 * 1024 } // 1 MiB
    }
}

/// Double-buffering transfer scheduler.
///
/// Alternates between two device buffers ("slots") so that the host can
/// prepare the next batch of data while the GPU processes the current one.
/// In the CPU reference implementation overlap is simulated by tracking
/// which slot is "active" and providing correct round-trip semantics.
#[derive(Debug)]
pub struct TransferScheduler {
    slot_a: DeviceBuffer,
    slot_b: DeviceBuffer,
    active: Slot,
    config: SchedulerConfig,
    transfers_completed: u64,
    total_bytes: u64,
}

impl TransferScheduler {
    /// Create a new scheduler with two slots of `config.slot_size` bytes.
    pub fn new(config: SchedulerConfig) -> TransferResult<Self> {
        if config.slot_size == 0 {
            return Err(TransferError::InvalidConfig("slot_size must be > 0".into()));
        }
        Ok(Self {
            slot_a: DeviceBuffer::allocate(config.slot_size),
            slot_b: DeviceBuffer::allocate(config.slot_size),
            active: Slot::A,
            config,
            transfers_completed: 0,
            total_bytes: 0,
        })
    }

    /// Upload `data` into the *inactive* slot, then swap.
    /// Returns the number of bytes transferred.
    pub fn submit(&mut self, data: &[u8]) -> TransferResult<usize> {
        let dst = match self.active {
            Slot::A => &mut self.slot_b,
            Slot::B => &mut self.slot_a,
        };
        let n = HostToDevice::transfer(data, dst)?;
        self.active = self.active.flip();
        self.transfers_completed += 1;
        self.total_bytes += n as u64;
        Ok(n)
    }

    /// Read the contents of the currently active slot.
    pub fn read_active(&self) -> TransferResult<Vec<u8>> {
        let src = match self.active {
            Slot::A => &self.slot_a,
            Slot::B => &self.slot_b,
        };
        DeviceToHost::download(src)
    }

    /// Read the contents of the inactive (back) slot.
    pub fn read_inactive(&self) -> TransferResult<Vec<u8>> {
        let src = match self.active {
            Slot::A => &self.slot_b,
            Slot::B => &self.slot_a,
        };
        DeviceToHost::download(src)
    }

    /// Which slot index (0 = A, 1 = B) is currently active.
    pub fn active_slot_index(&self) -> usize {
        match self.active {
            Slot::A => 0,
            Slot::B => 1,
        }
    }

    pub fn slot_size(&self) -> usize {
        self.config.slot_size
    }

    pub fn transfers_completed(&self) -> u64 {
        self.transfers_completed
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

// ---------------------------------------------------------------------------
// TransferStats — aggregate bandwidth / latency / queue depth
// ---------------------------------------------------------------------------

/// A single recorded transfer event.
#[derive(Debug, Clone)]
pub struct TransferRecord {
    pub direction: TransferDirection,
    pub bytes: usize,
    pub duration: Duration,
}

impl TransferRecord {
    /// Bandwidth in GB/s, or 0.0 if duration is zero.
    pub fn bandwidth_gbps(&self) -> f64 {
        let ns = self.duration.as_nanos();
        if ns == 0 {
            return 0.0;
        }
        (self.bytes as f64) / (ns as f64) // bytes/ns ≡ GB/s
    }
}

/// Aggregate transfer statistics.
#[derive(Debug)]
pub struct TransferStats {
    records: Vec<TransferRecord>,
    pending_count: u64,
}

impl TransferStats {
    pub fn new() -> Self {
        Self { records: Vec::new(), pending_count: 0 }
    }

    /// Record a completed transfer.
    pub fn record(&mut self, direction: TransferDirection, bytes: usize, duration: Duration) {
        self.records.push(TransferRecord { direction, bytes, duration });
    }

    /// Increment the number of currently-in-flight transfers.
    pub fn inc_pending(&mut self) {
        self.pending_count += 1;
    }

    /// Decrement the number of currently-in-flight transfers.
    pub fn dec_pending(&mut self) {
        self.pending_count = self.pending_count.saturating_sub(1);
    }

    /// Current queue depth (in-flight transfers).
    pub fn queue_depth(&self) -> u64 {
        self.pending_count
    }

    /// Total number of completed transfers.
    pub fn completed_count(&self) -> usize {
        self.records.len()
    }

    /// Total bytes across all completed transfers.
    pub fn total_bytes(&self) -> usize {
        self.records.iter().map(|r| r.bytes).sum()
    }

    /// Total duration across all completed transfers.
    pub fn total_duration(&self) -> Duration {
        self.records.iter().map(|r| r.duration).sum()
    }

    /// Average bandwidth in GB/s (or 0.0 when empty).
    pub fn avg_bandwidth_gbps(&self) -> f64 {
        let ns = self.total_duration().as_nanos();
        if ns == 0 {
            return 0.0;
        }
        (self.total_bytes() as f64) / (ns as f64)
    }

    /// Peak bandwidth across any single transfer (GB/s).
    pub fn peak_bandwidth_gbps(&self) -> f64 {
        self.records.iter().map(|r| r.bandwidth_gbps()).fold(0.0_f64, f64::max)
    }

    /// Average latency per transfer, or zero when empty.
    pub fn avg_latency(&self) -> Duration {
        let n = self.records.len() as u64;
        if n == 0 {
            return Duration::ZERO;
        }
        self.total_duration() / n as u32
    }

    /// Filtered stats for a given direction.
    pub fn stats_for_direction(&self, dir: TransferDirection) -> DirectionStats {
        let mut total_bytes: usize = 0;
        let mut count: usize = 0;
        let mut total_ns: u128 = 0;
        let mut peak_bw: f64 = 0.0;
        for r in &self.records {
            if r.direction == dir {
                total_bytes += r.bytes;
                count += 1;
                total_ns += r.duration.as_nanos();
                let bw = r.bandwidth_gbps();
                if bw > peak_bw {
                    peak_bw = bw;
                }
            }
        }
        DirectionStats {
            direction: dir,
            total_bytes,
            count,
            total_ns,
            peak_bandwidth_gbps: peak_bw,
        }
    }

    /// Clear all records and reset counters.
    pub fn reset(&mut self) {
        self.records.clear();
        self.pending_count = 0;
    }

    /// Get all records (for inspection).
    pub fn records(&self) -> &[TransferRecord] {
        &self.records
    }
}

impl Default for TransferStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-direction aggregate.
#[derive(Debug, Clone)]
pub struct DirectionStats {
    pub direction: TransferDirection,
    pub total_bytes: usize,
    pub count: usize,
    pub total_ns: u128,
    pub peak_bandwidth_gbps: f64,
}

impl DirectionStats {
    pub fn avg_bandwidth_gbps(&self) -> f64 {
        if self.total_ns == 0 {
            return 0.0;
        }
        (self.total_bytes as f64) / (self.total_ns as f64)
    }
}

// ---------------------------------------------------------------------------
// ZeroCopyBuffer — shared host-device memory
// ---------------------------------------------------------------------------

/// Simulated zero-copy (shared) buffer that is accessible from both host
/// and device without explicit transfers.
///
/// On hardware that supports `CL_MEM_USE_HOST_PTR` / SVM, reads and writes
/// are visible to both sides.  The CPU reference simply wraps a `Vec<u8>`.
#[derive(Debug, Clone)]
pub struct ZeroCopyBuffer {
    id: u64,
    data: Vec<u8>,
    mapped: bool,
}

impl ZeroCopyBuffer {
    /// Allocate a shared buffer of `size` bytes.
    pub fn allocate(size: usize) -> TransferResult<Self> {
        Ok(Self { id: next_id(), data: vec![0u8; size], mapped: true })
    }

    /// Create from existing host data.
    pub fn from_host_data(data: &[u8]) -> TransferResult<Self> {
        Ok(Self { id: next_id(), data: data.to_vec(), mapped: true })
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Read the shared buffer as a byte slice (host side).
    pub fn host_read(&self) -> TransferResult<&[u8]> {
        if !self.mapped {
            return Err(TransferError::ZeroCopyUnavailable);
        }
        Ok(&self.data)
    }

    /// Write into the shared buffer from host side.
    pub fn host_write(&mut self, offset: usize, src: &[u8]) -> TransferResult<usize> {
        if !self.mapped {
            return Err(TransferError::ZeroCopyUnavailable);
        }
        let end = offset.checked_add(src.len()).ok_or(TransferError::DestinationTooSmall {
            required: usize::MAX,
            available: self.data.len(),
        })?;
        if end > self.data.len() {
            return Err(TransferError::DestinationTooSmall {
                required: end,
                available: self.data.len(),
            });
        }
        self.data[offset..end].copy_from_slice(src);
        Ok(src.len())
    }

    /// Read from device perspective (same memory in CPU reference).
    pub fn device_read(&self) -> TransferResult<&[u8]> {
        self.host_read()
    }

    /// Write from device perspective.
    pub fn device_write(&mut self, offset: usize, src: &[u8]) -> TransferResult<usize> {
        self.host_write(offset, src)
    }

    /// Unmap the buffer, preventing further host access.
    pub fn unmap(&mut self) {
        self.mapped = false;
    }

    /// Re-map the buffer for host access.
    pub fn remap(&mut self) {
        self.mapped = true;
    }

    /// Copy the zero-copy buffer contents into a standard DeviceBuffer.
    pub fn to_device_buffer(&self) -> TransferResult<DeviceBuffer> {
        if !self.mapped {
            return Err(TransferError::ZeroCopyUnavailable);
        }
        Ok(DeviceBuffer::from_data(&self.data))
    }
}

// ---------------------------------------------------------------------------
// TransferQueue — FIFO of pending transfer descriptions
// ---------------------------------------------------------------------------

/// Describes a pending transfer to be executed later.
#[derive(Debug, Clone)]
pub struct TransferRequest {
    pub direction: TransferDirection,
    pub data: Vec<u8>,
    pub dst_offset: usize,
}

/// A simple FIFO queue of transfer requests.
#[derive(Debug)]
pub struct TransferQueue {
    queue: VecDeque<TransferRequest>,
    max_depth: usize,
}

impl TransferQueue {
    pub fn new(max_depth: usize) -> Self {
        Self { queue: VecDeque::with_capacity(max_depth), max_depth }
    }

    /// Enqueue a transfer request.
    pub fn enqueue(&mut self, req: TransferRequest) -> TransferResult<()> {
        if self.queue.len() >= self.max_depth {
            return Err(TransferError::StagingFull { capacity: self.max_depth });
        }
        self.queue.push_back(req);
        Ok(())
    }

    /// Dequeue the next transfer request.
    pub fn dequeue(&mut self) -> Option<TransferRequest> {
        self.queue.pop_front()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.max_depth
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Drain all requests into a `Vec`.
    pub fn drain_all(&mut self) -> Vec<TransferRequest> {
        self.queue.drain(..).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── H2D + D2H round-trip ───────────────────────────────────────────

    #[test]
    fn h2d_d2h_roundtrip_basic() {
        let src = b"hello gpu";
        let mut dev = DeviceBuffer::allocate(64);
        HostToDevice::transfer(src, &mut dev).unwrap();
        let back = DeviceToHost::download(&dev).unwrap();
        assert_eq!(&back[..src.len()], src);
    }

    #[test]
    fn h2d_d2h_roundtrip_exact_size() {
        let data: Vec<u8> = (0..=255).collect();
        let mut dev = DeviceBuffer::allocate(256);
        HostToDevice::transfer(&data, &mut dev).unwrap();
        let back = DeviceToHost::download(&dev).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn h2d_d2h_roundtrip_large() {
        let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
        let mut dev = DeviceBuffer::allocate(10_000);
        HostToDevice::transfer(&data, &mut dev).unwrap();
        let back = DeviceToHost::download(&dev).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn h2d_upload_creates_exact_buffer() {
        let data = vec![42u8; 128];
        let dev = HostToDevice::upload(&data);
        assert_eq!(dev.len(), 128);
        assert_eq!(dev.as_bytes(), &data[..]);
    }

    #[test]
    fn h2d_transfer_range_roundtrip() {
        let src = vec![0xAA; 32];
        let mut dev = DeviceBuffer::allocate(64);
        HostToDevice::transfer_range(&src, &mut dev, 16, 32).unwrap();
        let back = DeviceToHost::download(&dev).unwrap();
        assert_eq!(&back[16..48], &src[..]);
        // Untouched regions remain zero.
        assert!(back[..16].iter().all(|&b| b == 0));
    }

    #[test]
    fn h2d_d2h_roundtrip_all_byte_values() {
        let data: Vec<u8> = (0..=255).collect();
        let dev = HostToDevice::upload(&data);
        let back = DeviceToHost::download(&dev).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn h2d_transfer_to_pre_allocated_partial() {
        let mut dev = DeviceBuffer::allocate(64);
        let small = [1u8, 2, 3, 4];
        let n = HostToDevice::transfer(&small, &mut dev).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&dev.as_bytes()[..4], &small);
        assert_eq!(dev.as_bytes()[4], 0); // rest untouched
    }

    // ── D2H download range ─────────────────────────────────────────────

    #[test]
    fn d2h_download_range() {
        let data: Vec<u8> = (0..100).collect();
        let dev = DeviceBuffer::from_data(&data);
        let sub = DeviceToHost::download_range(&dev, 10, 20).unwrap();
        assert_eq!(sub, &data[10..30]);
    }

    #[test]
    fn d2h_transfer_into_existing_buffer() {
        let dev = DeviceBuffer::from_data(&[5u8; 8]);
        let mut host = vec![0u8; 16];
        let n = DeviceToHost::transfer(&dev, &mut host).unwrap();
        assert_eq!(n, 8);
        assert_eq!(&host[..8], &[5u8; 8]);
    }

    // ── Error cases ────────────────────────────────────────────────────

    #[test]
    fn h2d_destination_too_small() {
        let src = vec![0u8; 128];
        let mut dev = DeviceBuffer::allocate(64);
        let err = HostToDevice::transfer(&src, &mut dev).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    #[test]
    fn d2h_destination_too_small() {
        let dev = DeviceBuffer::from_data(&[0u8; 64]);
        let mut host = vec![0u8; 32];
        let err = DeviceToHost::transfer(&dev, &mut host).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    #[test]
    fn h2d_transfer_range_source_too_small() {
        let src = vec![0u8; 4];
        let mut dev = DeviceBuffer::allocate(64);
        let err = HostToDevice::transfer_range(&src, &mut dev, 0, 10).unwrap_err();
        assert!(matches!(err, TransferError::SourceTooSmall { .. }));
    }

    #[test]
    fn h2d_transfer_range_dst_overflow() {
        let src = vec![0u8; 32];
        let mut dev = DeviceBuffer::allocate(16);
        let err = HostToDevice::transfer_range(&src, &mut dev, 0, 32).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    #[test]
    fn d2h_download_range_out_of_bounds() {
        let dev = DeviceBuffer::from_data(&[0u8; 32]);
        let err = DeviceToHost::download_range(&dev, 30, 10).unwrap_err();
        assert!(matches!(err, TransferError::SourceTooSmall { .. }));
    }

    #[test]
    fn transfer_to_freed_buffer_fails() {
        let mut dev = DeviceBuffer::allocate(64);
        dev.free();
        let err = HostToDevice::transfer(&[1u8], &mut dev).unwrap_err();
        assert!(matches!(err, TransferError::BufferFreed));
    }

    #[test]
    fn download_from_freed_buffer_fails() {
        let mut dev = DeviceBuffer::from_data(&[1u8; 8]);
        dev.free();
        let err = DeviceToHost::download(&dev).unwrap_err();
        assert!(matches!(err, TransferError::BufferFreed));
    }

    // ── Zero-size transfers ────────────────────────────────────────────

    #[test]
    fn zero_size_h2d_transfer() {
        let mut dev = DeviceBuffer::allocate(64);
        let n = HostToDevice::transfer(&[], &mut dev).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn zero_size_d2h_download() {
        let dev = DeviceBuffer::allocate(0);
        let data = DeviceToHost::download(&dev).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn zero_size_device_buffer() {
        let dev = DeviceBuffer::allocate(0);
        assert!(dev.is_empty());
        assert_eq!(dev.len(), 0);
    }

    #[test]
    fn zero_size_d2d_copy() {
        let src = DeviceBuffer::allocate(0);
        let mut dst = DeviceBuffer::allocate(0);
        let n = DeviceToDevice::copy(&src, &mut dst).unwrap();
        assert_eq!(n, 0);
    }

    // ── D2D copy ───────────────────────────────────────────────────────

    #[test]
    fn d2d_full_copy() {
        let data: Vec<u8> = (0..64).collect();
        let src = DeviceBuffer::from_data(&data);
        let mut dst = DeviceBuffer::allocate(64);
        let n = DeviceToDevice::copy(&src, &mut dst).unwrap();
        assert_eq!(n, 64);
        assert_eq!(dst.as_bytes(), src.as_bytes());
    }

    #[test]
    fn d2d_copy_range() {
        let data: Vec<u8> = (0..128).collect();
        let src = DeviceBuffer::from_data(&data);
        let mut dst = DeviceBuffer::allocate(128);
        DeviceToDevice::copy_range(&src, 32, &mut dst, 64, 32).unwrap();
        assert_eq!(&dst.as_bytes()[64..96], &data[32..64]);
        // Other regions untouched.
        assert!(dst.as_bytes()[..64].iter().all(|&b| b == 0));
    }

    #[test]
    fn d2d_duplicate() {
        let data = vec![0xBB; 48];
        let src = DeviceBuffer::from_data(&data);
        let dup = DeviceToDevice::duplicate(&src).unwrap();
        assert_eq!(dup.as_bytes(), src.as_bytes());
        assert_ne!(dup.id(), src.id()); // distinct allocation
    }

    #[test]
    fn d2d_copy_dst_too_small() {
        let src = DeviceBuffer::from_data(&[0u8; 64]);
        let mut dst = DeviceBuffer::allocate(32);
        let err = DeviceToDevice::copy(&src, &mut dst).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    #[test]
    fn d2d_copy_range_src_oob() {
        let src = DeviceBuffer::from_data(&[0u8; 16]);
        let mut dst = DeviceBuffer::allocate(32);
        let err = DeviceToDevice::copy_range(&src, 8, &mut dst, 0, 16).unwrap_err();
        assert!(matches!(err, TransferError::SourceTooSmall { .. }));
    }

    #[test]
    fn d2d_copy_range_dst_oob() {
        let src = DeviceBuffer::from_data(&[0u8; 32]);
        let mut dst = DeviceBuffer::allocate(16);
        let err = DeviceToDevice::copy_range(&src, 0, &mut dst, 8, 16).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    #[test]
    fn d2d_copy_from_freed_fails() {
        let mut src = DeviceBuffer::from_data(&[1u8; 16]);
        let mut dst = DeviceBuffer::allocate(16);
        src.free();
        let err = DeviceToDevice::copy(&src, &mut dst).unwrap_err();
        assert!(matches!(err, TransferError::BufferFreed));
    }

    #[test]
    fn d2d_copy_to_freed_fails() {
        let src = DeviceBuffer::from_data(&[1u8; 16]);
        let mut dst = DeviceBuffer::allocate(16);
        dst.free();
        let err = DeviceToDevice::copy(&src, &mut dst).unwrap_err();
        assert!(matches!(err, TransferError::BufferFreed));
    }

    // ── PinnedBuffer ───────────────────────────────────────────────────

    #[test]
    fn pinned_buffer_roundtrip() {
        let data = vec![0xCC; 64];
        let pinned = PinnedBuffer::from_data(&data);
        assert!(pinned.is_pinned());
        assert_eq!(pinned.len(), 64);
        assert_eq!(pinned.as_bytes(), &data[..]);
    }

    #[test]
    fn pinned_upload_download() {
        let pinned = PinnedBuffer::from_data(&[1, 2, 3, 4]);
        let mut dev = DeviceBuffer::allocate(8);
        pinned.upload_to(&mut dev).unwrap();
        let mut pinned2 = PinnedBuffer::allocate(8);
        pinned2.download_from(&dev).unwrap();
        assert_eq!(&pinned2.as_bytes()[..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn pinned_unpin_flag() {
        let mut p = PinnedBuffer::allocate(16);
        assert!(p.is_pinned());
        p.unpin();
        assert!(!p.is_pinned());
    }

    #[test]
    fn pinned_bandwidth_multiplier_positive() {
        assert!(PinnedBuffer::bandwidth_multiplier() > 1.0);
    }

    #[test]
    fn pinned_zero_size() {
        let p = PinnedBuffer::allocate(0);
        assert!(p.is_empty());
    }

    #[test]
    fn pinned_download_from_freed_fails() {
        let mut dev = DeviceBuffer::from_data(&[0u8; 16]);
        dev.free();
        let mut pinned = PinnedBuffer::allocate(16);
        let err = pinned.download_from(&dev).unwrap_err();
        assert!(matches!(err, TransferError::BufferFreed));
    }

    #[test]
    fn pinned_download_dst_too_small() {
        let dev = DeviceBuffer::from_data(&[0u8; 32]);
        let mut pinned = PinnedBuffer::allocate(16);
        let err = pinned.download_from(&dev).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    // ── AsyncTransfer ──────────────────────────────────────────────────

    #[test]
    fn async_h2d_completes_immediately() {
        let data = vec![7u8; 32];
        let mut dev = DeviceBuffer::allocate(32);
        let xfer = AsyncTransfer::host_to_device(&data, &mut dev).unwrap();
        assert!(xfer.is_complete());
        assert_eq!(xfer.status(), AsyncStatus::Complete);
        assert_eq!(xfer.bytes(), 32);
        assert_eq!(xfer.direction(), TransferDirection::HostToDevice);
    }

    #[test]
    fn async_d2h_returns_data() {
        let dev = DeviceBuffer::from_data(&[9u8; 16]);
        let (xfer, data) = AsyncTransfer::device_to_host(&dev).unwrap();
        assert!(xfer.is_complete());
        assert_eq!(data, vec![9u8; 16]);
    }

    #[test]
    fn async_d2d_completes() {
        let src = DeviceBuffer::from_data(&[3u8; 24]);
        let mut dst = DeviceBuffer::allocate(24);
        let xfer = AsyncTransfer::device_to_device(&src, &mut dst).unwrap();
        assert!(xfer.is_complete());
        assert_eq!(xfer.bytes(), 24);
        assert_eq!(dst.as_bytes(), src.as_bytes());
    }

    #[test]
    fn async_wait_succeeds() {
        let mut dev = DeviceBuffer::allocate(8);
        let mut xfer = AsyncTransfer::host_to_device(&[0u8; 8], &mut dev).unwrap();
        xfer.wait().unwrap();
        assert!(xfer.is_complete());
    }

    #[test]
    fn async_elapsed_is_finite() {
        let mut dev = DeviceBuffer::allocate(8);
        let xfer = AsyncTransfer::host_to_device(&[0u8; 8], &mut dev).unwrap();
        assert!(xfer.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn async_id_unique() {
        let mut d1 = DeviceBuffer::allocate(4);
        let mut d2 = DeviceBuffer::allocate(4);
        let t1 = AsyncTransfer::host_to_device(&[0u8; 4], &mut d1).unwrap();
        let t2 = AsyncTransfer::host_to_device(&[0u8; 4], &mut d2).unwrap();
        assert_ne!(t1.id(), t2.id());
    }

    // ── StagingBuffer ──────────────────────────────────────────────────

    #[test]
    fn staging_push_and_flush() {
        let mut staging = StagingBuffer::new(256);
        staging.push(&[1, 2, 3]).unwrap();
        staging.push(&[4, 5]).unwrap();
        assert_eq!(staging.region_count(), 2);
        assert_eq!(staging.staged_bytes(), 5);

        let mut dev = DeviceBuffer::allocate(256);
        let n = staging.flush_to(&mut dev, 0).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&dev.as_bytes()[..5], &[1, 2, 3, 4, 5]);
        assert_eq!(staging.staged_bytes(), 0);
        assert_eq!(staging.flush_count(), 1);
    }

    #[test]
    fn staging_overflow_rejected() {
        let mut staging = StagingBuffer::new(8);
        staging.push(&[0u8; 6]).unwrap();
        let err = staging.push(&[0u8; 4]).unwrap_err();
        assert!(matches!(err, TransferError::StagingFull { .. }));
    }

    #[test]
    fn staging_clear_discards_data() {
        let mut staging = StagingBuffer::new(64);
        staging.push(&[1, 2, 3]).unwrap();
        staging.clear();
        assert_eq!(staging.staged_bytes(), 0);
        assert_eq!(staging.region_count(), 0);
    }

    #[test]
    fn staging_flush_empty_is_noop() {
        let mut staging = StagingBuffer::new(64);
        let mut dev = DeviceBuffer::allocate(64);
        let n = staging.flush_to(&mut dev, 0).unwrap();
        assert_eq!(n, 0);
        assert_eq!(staging.flush_count(), 0);
    }

    #[test]
    fn staging_remaining_capacity() {
        let mut staging = StagingBuffer::new(100);
        staging.push(&[0u8; 30]).unwrap();
        assert_eq!(staging.remaining(), 70);
        assert_eq!(staging.capacity(), 100);
    }

    #[test]
    fn staging_regions_metadata() {
        let mut staging = StagingBuffer::new(256);
        staging.push(&[0u8; 10]).unwrap();
        staging.push(&[0u8; 20]).unwrap();
        let regions = staging.regions();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0], (0, 10));
        assert_eq!(regions[1], (10, 20));
    }

    #[test]
    fn staging_multiple_flushes() {
        let mut staging = StagingBuffer::new(64);
        let mut dev = DeviceBuffer::allocate(128);
        staging.push(&[0xAA; 16]).unwrap();
        staging.flush_to(&mut dev, 0).unwrap();
        staging.push(&[0xBB; 16]).unwrap();
        staging.flush_to(&mut dev, 64).unwrap();
        assert_eq!(staging.flush_count(), 2);
        assert_eq!(&dev.as_bytes()[..16], &[0xAA; 16]);
        assert_eq!(&dev.as_bytes()[64..80], &[0xBB; 16]);
    }

    #[test]
    fn staging_flush_at_offset() {
        let mut staging = StagingBuffer::new(64);
        staging.push(&[0xFF; 8]).unwrap();
        let mut dev = DeviceBuffer::allocate(64);
        staging.flush_to(&mut dev, 32).unwrap();
        assert_eq!(&dev.as_bytes()[32..40], &[0xFF; 8]);
        assert!(dev.as_bytes()[..32].iter().all(|&b| b == 0));
    }

    // ── TransferScheduler (double-buffering) ───────────────────────────

    #[test]
    fn scheduler_alternates_slots() {
        let sched = TransferScheduler::new(SchedulerConfig { slot_size: 64 }).unwrap();
        assert_eq!(sched.active_slot_index(), 0);
    }

    #[test]
    fn scheduler_submit_and_read() {
        let mut sched = TransferScheduler::new(SchedulerConfig { slot_size: 64 }).unwrap();
        let data = vec![0xDD; 64];
        sched.submit(&data).unwrap();
        // After submit, active flipped — the data is now in the active slot.
        let active = sched.read_active().unwrap();
        assert_eq!(&active[..], &data[..]);
    }

    #[test]
    fn scheduler_double_buffer_swap() {
        let mut sched = TransferScheduler::new(SchedulerConfig { slot_size: 32 }).unwrap();
        let batch_a = vec![0xAA; 32];
        let batch_b = vec![0xBB; 32];

        sched.submit(&batch_a).unwrap();
        assert_eq!(sched.active_slot_index(), 1); // flipped to B
        assert_eq!(sched.read_active().unwrap(), batch_a);

        sched.submit(&batch_b).unwrap();
        assert_eq!(sched.active_slot_index(), 0); // flipped back to A
        assert_eq!(sched.read_active().unwrap(), batch_b);

        // Inactive slot still has batch_a from the first submit
        assert_eq!(sched.read_inactive().unwrap(), batch_a);
    }

    #[test]
    fn scheduler_tracks_stats() {
        let mut sched = TransferScheduler::new(SchedulerConfig { slot_size: 128 }).unwrap();
        sched.submit(&[0u8; 100]).unwrap();
        sched.submit(&[0u8; 128]).unwrap();
        assert_eq!(sched.transfers_completed(), 2);
        assert_eq!(sched.total_bytes(), 228);
    }

    #[test]
    fn scheduler_zero_slot_size_rejected() {
        let err = TransferScheduler::new(SchedulerConfig { slot_size: 0 }).unwrap_err();
        assert!(matches!(err, TransferError::InvalidConfig(_)));
    }

    #[test]
    fn scheduler_data_too_large() {
        let mut sched = TransferScheduler::new(SchedulerConfig { slot_size: 16 }).unwrap();
        let err = sched.submit(&[0u8; 32]).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    // ── TransferStats ──────────────────────────────────────────────────

    #[test]
    fn stats_empty() {
        let stats = TransferStats::new();
        assert_eq!(stats.completed_count(), 0);
        assert_eq!(stats.total_bytes(), 0);
        assert_eq!(stats.avg_bandwidth_gbps(), 0.0);
        assert_eq!(stats.peak_bandwidth_gbps(), 0.0);
        assert_eq!(stats.queue_depth(), 0);
    }

    #[test]
    fn stats_record_and_query() {
        let mut stats = TransferStats::new();
        stats.record(TransferDirection::HostToDevice, 1_000_000, Duration::from_micros(100));
        assert_eq!(stats.completed_count(), 1);
        assert_eq!(stats.total_bytes(), 1_000_000);
        assert!(stats.avg_bandwidth_gbps() > 0.0);
    }

    #[test]
    fn stats_peak_bandwidth() {
        let mut stats = TransferStats::new();
        // Slow transfer
        stats.record(TransferDirection::HostToDevice, 1000, Duration::from_millis(10));
        // Fast transfer
        stats.record(TransferDirection::HostToDevice, 1_000_000, Duration::from_micros(50));
        assert!(stats.peak_bandwidth_gbps() > stats.avg_bandwidth_gbps());
    }

    #[test]
    fn stats_queue_depth() {
        let mut stats = TransferStats::new();
        stats.inc_pending();
        stats.inc_pending();
        assert_eq!(stats.queue_depth(), 2);
        stats.dec_pending();
        assert_eq!(stats.queue_depth(), 1);
        stats.dec_pending();
        stats.dec_pending(); // saturates
        assert_eq!(stats.queue_depth(), 0);
    }

    #[test]
    fn stats_per_direction() {
        let mut stats = TransferStats::new();
        stats.record(TransferDirection::HostToDevice, 100, Duration::from_nanos(50));
        stats.record(TransferDirection::DeviceToHost, 200, Duration::from_nanos(80));
        stats.record(TransferDirection::HostToDevice, 300, Duration::from_nanos(60));

        let h2d = stats.stats_for_direction(TransferDirection::HostToDevice);
        assert_eq!(h2d.count, 2);
        assert_eq!(h2d.total_bytes, 400);

        let d2h = stats.stats_for_direction(TransferDirection::DeviceToHost);
        assert_eq!(d2h.count, 1);
        assert_eq!(d2h.total_bytes, 200);
    }

    #[test]
    fn stats_reset() {
        let mut stats = TransferStats::new();
        stats.record(TransferDirection::HostToDevice, 1000, Duration::from_nanos(10));
        stats.inc_pending();
        stats.reset();
        assert_eq!(stats.completed_count(), 0);
        assert_eq!(stats.queue_depth(), 0);
    }

    #[test]
    fn stats_avg_latency() {
        let mut stats = TransferStats::new();
        stats.record(TransferDirection::HostToDevice, 100, Duration::from_millis(10));
        stats.record(TransferDirection::DeviceToHost, 200, Duration::from_millis(30));
        assert_eq!(stats.avg_latency(), Duration::from_millis(20));
    }

    // ── ZeroCopyBuffer ─────────────────────────────────────────────────

    #[test]
    fn zero_copy_host_write_read() {
        let mut zc = ZeroCopyBuffer::allocate(64).unwrap();
        zc.host_write(0, &[1, 2, 3, 4]).unwrap();
        let data = zc.host_read().unwrap();
        assert_eq!(&data[..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn zero_copy_device_write_read() {
        let mut zc = ZeroCopyBuffer::allocate(32).unwrap();
        zc.device_write(8, &[0xFF; 4]).unwrap();
        let data = zc.device_read().unwrap();
        assert_eq!(&data[8..12], &[0xFF; 4]);
    }

    #[test]
    fn zero_copy_shared_visibility() {
        let mut zc = ZeroCopyBuffer::allocate(16).unwrap();
        zc.host_write(0, &[42]).unwrap();
        let dev_view = zc.device_read().unwrap();
        assert_eq!(dev_view[0], 42);
    }

    #[test]
    fn zero_copy_from_host_data() {
        let data = vec![10u8; 20];
        let zc = ZeroCopyBuffer::from_host_data(&data).unwrap();
        assert_eq!(zc.host_read().unwrap(), &data[..]);
    }

    #[test]
    fn zero_copy_unmap_blocks_access() {
        let mut zc = ZeroCopyBuffer::allocate(16).unwrap();
        zc.unmap();
        assert!(!zc.is_mapped());
        let err = zc.host_read().unwrap_err();
        assert!(matches!(err, TransferError::ZeroCopyUnavailable));
        let err = zc.host_write(0, &[1]).unwrap_err();
        assert!(matches!(err, TransferError::ZeroCopyUnavailable));
    }

    #[test]
    fn zero_copy_remap_restores_access() {
        let mut zc = ZeroCopyBuffer::allocate(16).unwrap();
        zc.host_write(0, &[5]).unwrap();
        zc.unmap();
        zc.remap();
        assert!(zc.is_mapped());
        let data = zc.host_read().unwrap();
        assert_eq!(data[0], 5);
    }

    #[test]
    fn zero_copy_write_out_of_bounds() {
        let mut zc = ZeroCopyBuffer::allocate(8).unwrap();
        let err = zc.host_write(4, &[0u8; 8]).unwrap_err();
        assert!(matches!(err, TransferError::DestinationTooSmall { .. }));
    }

    #[test]
    fn zero_copy_to_device_buffer() {
        let data = vec![0xEE; 32];
        let zc = ZeroCopyBuffer::from_host_data(&data).unwrap();
        let dev = zc.to_device_buffer().unwrap();
        assert_eq!(dev.as_bytes(), &data[..]);
    }

    #[test]
    fn zero_copy_zero_size() {
        let zc = ZeroCopyBuffer::allocate(0).unwrap();
        assert!(zc.is_empty());
        assert_eq!(zc.len(), 0);
    }

    #[test]
    fn zero_copy_to_device_buffer_when_unmapped_fails() {
        let mut zc = ZeroCopyBuffer::allocate(16).unwrap();
        zc.unmap();
        let err = zc.to_device_buffer().unwrap_err();
        assert!(matches!(err, TransferError::ZeroCopyUnavailable));
    }

    // ── TransferQueue ──────────────────────────────────────────────────

    #[test]
    fn queue_enqueue_dequeue() {
        let mut q = TransferQueue::new(4);
        q.enqueue(TransferRequest {
            direction: TransferDirection::HostToDevice,
            data: vec![1, 2, 3],
            dst_offset: 0,
        })
        .unwrap();
        assert_eq!(q.len(), 1);
        let req = q.dequeue().unwrap();
        assert_eq!(req.data, vec![1, 2, 3]);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_full_rejected() {
        let mut q = TransferQueue::new(2);
        q.enqueue(TransferRequest {
            direction: TransferDirection::HostToDevice,
            data: vec![],
            dst_offset: 0,
        })
        .unwrap();
        q.enqueue(TransferRequest {
            direction: TransferDirection::HostToDevice,
            data: vec![],
            dst_offset: 0,
        })
        .unwrap();
        let err = q
            .enqueue(TransferRequest {
                direction: TransferDirection::HostToDevice,
                data: vec![],
                dst_offset: 0,
            })
            .unwrap_err();
        assert!(matches!(err, TransferError::StagingFull { .. }));
    }

    #[test]
    fn queue_drain_all() {
        let mut q = TransferQueue::new(8);
        for i in 0..3 {
            q.enqueue(TransferRequest {
                direction: TransferDirection::DeviceToHost,
                data: vec![i],
                dst_offset: 0,
            })
            .unwrap();
        }
        let drained = q.drain_all();
        assert_eq!(drained.len(), 3);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_fifo_ordering() {
        let mut q = TransferQueue::new(8);
        q.enqueue(TransferRequest {
            direction: TransferDirection::HostToDevice,
            data: vec![1],
            dst_offset: 0,
        })
        .unwrap();
        q.enqueue(TransferRequest {
            direction: TransferDirection::HostToDevice,
            data: vec![2],
            dst_offset: 0,
        })
        .unwrap();
        assert_eq!(q.dequeue().unwrap().data, vec![1]);
        assert_eq!(q.dequeue().unwrap().data, vec![2]);
    }

    // ── DeviceBuffer misc ──────────────────────────────────────────────

    #[test]
    fn device_buffer_unique_ids() {
        let a = DeviceBuffer::allocate(8);
        let b = DeviceBuffer::allocate(8);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn device_buffer_free_clears_data() {
        let mut dev = DeviceBuffer::allocate(64);
        dev.free();
        assert!(dev.is_freed());
        assert!(dev.is_empty());
    }

    // ── TransferDirection display ──────────────────────────────────────

    #[test]
    fn direction_display() {
        assert_eq!(format!("{}", TransferDirection::HostToDevice), "H2D");
        assert_eq!(format!("{}", TransferDirection::DeviceToHost), "D2H");
        assert_eq!(format!("{}", TransferDirection::DeviceToDevice), "D2D");
    }

    // ── TransferError display ──────────────────────────────────────────

    #[test]
    fn error_display_messages() {
        let err = TransferError::SourceTooSmall { required: 100, available: 50 };
        assert!(format!("{err}").contains("100"));

        let err = TransferError::NotReady;
        assert!(format!("{err}").contains("not yet complete"));

        let err = TransferError::BufferFreed;
        assert!(format!("{err}").contains("freed"));

        let err = TransferError::ZeroCopyUnavailable;
        assert!(format!("{err}").contains("unavailable"));
    }

    // ── Property: transfer preserves data exactly ──────────────────────

    #[test]
    fn property_h2d_d2h_identity_small_sizes() {
        for size in [0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256] {
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let dev = HostToDevice::upload(&data);
            let back = DeviceToHost::download(&dev).unwrap();
            assert_eq!(back, data, "round-trip failed for size {size}");
        }
    }

    #[test]
    fn property_d2d_preserves_data() {
        for size in [0, 1, 64, 256, 1024] {
            let data: Vec<u8> = (0..size).map(|i| (i * 7 % 256) as u8).collect();
            let src = DeviceBuffer::from_data(&data);
            let dup = DeviceToDevice::duplicate(&src).unwrap();
            assert_eq!(dup.as_bytes(), src.as_bytes(), "D2D dup failed for size {size}");
        }
    }

    #[test]
    fn property_staging_preserves_order() {
        let mut staging = StagingBuffer::new(1024);
        let chunks: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8; 10]).collect();
        for chunk in &chunks {
            staging.push(chunk).unwrap();
        }
        let expected: Vec<u8> = chunks.into_iter().flatten().collect();
        assert_eq!(staging.as_bytes(), &expected[..]);
    }

    #[test]
    fn property_scheduler_never_loses_data() {
        let mut sched = TransferScheduler::new(SchedulerConfig { slot_size: 64 }).unwrap();
        let batches: Vec<Vec<u8>> = (0..8).map(|i| vec![i; 64]).collect();
        for batch in &batches {
            sched.submit(batch).unwrap();
            let active = sched.read_active().unwrap();
            assert_eq!(&active[..], &batch[..]);
        }
    }

    // ── Max-size transfer ──────────────────────────────────────────────

    #[test]
    fn max_size_transfer_64kb() {
        let size = 64 * 1024;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let dev = HostToDevice::upload(&data);
        let back = DeviceToHost::download(&dev).unwrap();
        assert_eq!(back, data);
    }

    // ── Pinned vs unpinned bandwidth comparison ────────────────────────

    #[test]
    fn pinned_vs_unpinned_bandwidth_ratio() {
        // CPU reference: the multiplier should reflect that pinned is faster.
        let mult = PinnedBuffer::bandwidth_multiplier();
        assert!(mult >= 1.5, "expected ≥1.5× for pinned, got {mult}");
        assert!(mult <= 4.0, "expected ≤4.0× for pinned, got {mult}");
    }

    // ── DirectionStats avg_bandwidth ───────────────────────────────────

    #[test]
    fn direction_stats_avg_bandwidth() {
        let mut stats = TransferStats::new();
        stats.record(TransferDirection::DeviceToDevice, 500, Duration::from_nanos(100));
        let d2d = stats.stats_for_direction(TransferDirection::DeviceToDevice);
        // 500 bytes / 100 ns = 5 GB/s
        let bw = d2d.avg_bandwidth_gbps();
        assert!((bw - 5.0).abs() < 0.01, "expected ~5 GB/s, got {bw}");
    }

    #[test]
    fn direction_stats_empty() {
        let stats = TransferStats::new();
        let h2d = stats.stats_for_direction(TransferDirection::HostToDevice);
        assert_eq!(h2d.count, 0);
        assert_eq!(h2d.avg_bandwidth_gbps(), 0.0);
    }
}
