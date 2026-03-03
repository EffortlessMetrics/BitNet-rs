//! CUDA asynchronous memory copy with staged transfers and bandwidth optimization.
//!
//! # Overview
//!
//! Models async memcpy operations for overlapping data transfers with compute.
//! Provides staged transfer planning, bandwidth utilization, and copy scheduling
//! with priority queues and dependency tracking.
//!
//! Key components:
//!
//! - [`CopyDirection`] — transfer direction (H2D, D2H, D2D, peer-to-peer)
//! - [`AsyncCopyOp`] — a single async memory copy operation
//! - [`StagedTransferPlan`] — pinned-memory staging with optional double buffering
//! - [`OverlapEstimator`] — estimate copy/compute overlap for pipelining
//! - [`BandwidthCalculator`] — PCIe and NVLink bandwidth utilization
//! - [`CopyScheduler`] — priority queue scheduler with dependency tracking
//! - [`PinnedMemoryManager`] — pinned (page-locked) host memory allocator
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations simulate transfers for testing on non-GPU hosts.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bitnet_common::{KernelError, Result};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_COPY_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_PINNED_ID: AtomicU64 = AtomicU64::new(1);

fn next_copy_id() -> u64 {
    NEXT_COPY_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_pinned_id() -> u64 {
    NEXT_PINNED_ID.fetch_add(1, Ordering::Relaxed)
}

// ── CopyDirection ────────────────────────────────────────────────────

/// Direction of an async memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CopyDirection {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Device to device (same GPU).
    DeviceToDevice,
    /// Peer-to-peer across GPUs.
    PeerToPeer { src_device: u32, dst_device: u32 },
}

impl fmt::Display for CopyDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H2D"),
            Self::DeviceToHost => write!(f, "D2H"),
            Self::DeviceToDevice => write!(f, "D2D"),
            Self::PeerToPeer { src_device, dst_device } => {
                write!(f, "P2P({src_device}->{dst_device})")
            }
        }
    }
}

// ── CopyPriority ─────────────────────────────────────────────────────

/// Priority level for a copy operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CopyPriority {
    /// Background pre-fetching — lowest priority.
    Background,
    /// Normal transfer priority.
    #[default]
    Normal,
    /// High priority — latency-sensitive.
    High,
    /// Critical — inference-blocking transfer.
    Critical,
}

impl CopyPriority {
    /// Numeric priority for ordering (higher = more urgent).
    pub fn as_numeric(self) -> u8 {
        match self {
            Self::Background => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

// ── CopyStatus ───────────────────────────────────────────────────────

/// Status of an async copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyStatus {
    /// Queued but not yet started.
    Pending,
    /// Transfer in progress.
    InProgress,
    /// Transfer completed successfully.
    Completed,
    /// Transfer failed.
    Failed,
    /// Transfer was cancelled.
    Cancelled,
}

// ── AsyncCopyOp ──────────────────────────────────────────────────────

/// Unique handle for an async copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CopyId(u64);

impl CopyId {
    fn next() -> Self {
        Self(next_copy_id())
    }
}

impl fmt::Display for CopyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "copy-{}", self.0)
    }
}

/// A single asynchronous memory copy operation.
#[derive(Debug, Clone)]
pub struct AsyncCopyOp {
    /// Unique identifier.
    pub id: CopyId,
    /// Transfer direction.
    pub direction: CopyDirection,
    /// Number of bytes to transfer.
    pub size_bytes: usize,
    /// Stream index this copy is assigned to.
    pub stream_index: u32,
    /// Priority level.
    pub priority: CopyPriority,
    /// Current status.
    pub status: CopyStatus,
    /// Whether this copy uses pinned (page-locked) host memory.
    pub uses_pinned_memory: bool,
    /// Optional label for profiling.
    pub label: String,
    /// Dependencies: IDs of ops that must complete before this one starts.
    pub depends_on: Vec<CopyId>,
    /// Estimated transfer duration (computed from bandwidth model).
    pub estimated_duration: Option<Duration>,
}

impl AsyncCopyOp {
    /// Create a new async copy operation.
    pub fn new(direction: CopyDirection, size_bytes: usize, label: impl Into<String>) -> Self {
        Self {
            id: CopyId::next(),
            direction,
            size_bytes,
            stream_index: 0,
            priority: CopyPriority::default(),
            status: CopyStatus::Pending,
            uses_pinned_memory: false,
            label: label.into(),
            depends_on: Vec::new(),
            estimated_duration: None,
        }
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: CopyPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set stream index.
    pub fn with_stream(mut self, stream_index: u32) -> Self {
        self.stream_index = stream_index;
        self
    }

    /// Mark as using pinned memory.
    pub fn with_pinned_memory(mut self) -> Self {
        self.uses_pinned_memory = true;
        self
    }

    /// Add a dependency on another copy op.
    pub fn depends_on(mut self, dep: CopyId) -> Self {
        self.depends_on.push(dep);
        self
    }
}

// ── Interconnect ─────────────────────────────────────────────────────

/// Type of interconnect between host and device or between devices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterconnectType {
    /// PCIe Gen3 x16 (≈15.75 GB/s per direction).
    PcieGen3,
    /// PCIe Gen4 x16 (≈31.5 GB/s per direction).
    PcieGen4,
    /// PCIe Gen5 x16 (≈63 GB/s per direction).
    PcieGen5,
    /// NVLink 2.0 (≈25 GB/s per sub-link, 6 links → 150 GB/s).
    NvLink2,
    /// NVLink 3.0 (≈25 GB/s per sub-link, 12 links → 300 GB/s).
    NvLink3,
    /// NVLink 4.0 (≈25 GB/s per sub-link, 18 links → 450 GB/s).
    NvLink4,
    /// Custom bandwidth specification.
    Custom { bandwidth_gbps: f64 },
}

impl InterconnectType {
    /// Theoretical peak unidirectional bandwidth in bytes per second.
    pub fn peak_bandwidth_bytes_per_sec(&self) -> f64 {
        match self {
            Self::PcieGen3 => 15.75 * 1e9,
            Self::PcieGen4 => 31.5 * 1e9,
            Self::PcieGen5 => 63.0 * 1e9,
            Self::NvLink2 => 150.0 * 1e9,
            Self::NvLink3 => 300.0 * 1e9,
            Self::NvLink4 => 450.0 * 1e9,
            Self::Custom { bandwidth_gbps } => bandwidth_gbps * 1e9,
        }
    }

    /// Whether this interconnect supports peer-to-peer transfers natively.
    pub fn supports_p2p(&self) -> bool {
        matches!(self, Self::NvLink2 | Self::NvLink3 | Self::NvLink4)
    }
}

// ── BandwidthCalculator ──────────────────────────────────────────────

/// Calculates bandwidth utilization and transfer times.
#[derive(Debug, Clone)]
pub struct BandwidthCalculator {
    /// Interconnect type for host↔device transfers.
    pub host_device_interconnect: InterconnectType,
    /// Interconnect type for peer-to-peer transfers.
    pub p2p_interconnect: Option<InterconnectType>,
    /// Achievable fraction of peak bandwidth (0.0–1.0).
    pub efficiency_factor: f64,
    /// Protocol overhead per transfer in bytes.
    pub overhead_bytes: usize,
    /// Minimum latency per transfer (PCIe round-trip, etc.).
    pub base_latency: Duration,
}

impl Default for BandwidthCalculator {
    fn default() -> Self {
        Self {
            host_device_interconnect: InterconnectType::PcieGen4,
            p2p_interconnect: None,
            efficiency_factor: 0.85,
            overhead_bytes: 64,
            base_latency: Duration::from_micros(5),
        }
    }
}

impl BandwidthCalculator {
    /// Create a calculator with the given interconnect type.
    pub fn new(interconnect: InterconnectType) -> Self {
        Self { host_device_interconnect: interconnect, ..Default::default() }
    }

    /// Set efficiency factor (clamped to 0.0–1.0).
    pub fn with_efficiency(mut self, factor: f64) -> Self {
        self.efficiency_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set peer-to-peer interconnect.
    pub fn with_p2p_interconnect(mut self, interconnect: InterconnectType) -> Self {
        self.p2p_interconnect = Some(interconnect);
        self
    }

    /// Set base latency.
    pub fn with_base_latency(mut self, latency: Duration) -> Self {
        self.base_latency = latency;
        self
    }

    /// Effective bandwidth for the given direction in bytes/sec.
    pub fn effective_bandwidth(&self, direction: CopyDirection) -> f64 {
        let peak = match direction {
            CopyDirection::PeerToPeer { .. } => self
                .p2p_interconnect
                .unwrap_or(self.host_device_interconnect)
                .peak_bandwidth_bytes_per_sec(),
            CopyDirection::DeviceToDevice => {
                // D2D on same GPU uses internal bandwidth — model as 2×
                self.host_device_interconnect.peak_bandwidth_bytes_per_sec() * 2.0
            }
            _ => self.host_device_interconnect.peak_bandwidth_bytes_per_sec(),
        };
        peak * self.efficiency_factor
    }

    /// Estimate transfer time for a given operation.
    pub fn estimate_transfer_time(&self, op: &AsyncCopyOp) -> Duration {
        let total_bytes = op.size_bytes + self.overhead_bytes;
        let bw = self.effective_bandwidth(op.direction);
        if bw <= 0.0 {
            return self.base_latency;
        }
        let transfer_secs = total_bytes as f64 / bw;
        self.base_latency + Duration::from_secs_f64(transfer_secs)
    }

    /// Bandwidth utilization given observed transfer time and bytes.
    pub fn utilization(
        &self,
        direction: CopyDirection,
        size_bytes: usize,
        observed: Duration,
    ) -> BandwidthUtilization {
        let peak = match direction {
            CopyDirection::PeerToPeer { .. } => self
                .p2p_interconnect
                .unwrap_or(self.host_device_interconnect)
                .peak_bandwidth_bytes_per_sec(),
            CopyDirection::DeviceToDevice => {
                self.host_device_interconnect.peak_bandwidth_bytes_per_sec() * 2.0
            }
            _ => self.host_device_interconnect.peak_bandwidth_bytes_per_sec(),
        };
        let secs = observed.as_secs_f64();
        let achieved_bw = if secs > 0.0 { size_bytes as f64 / secs } else { 0.0 };
        let ratio = if peak > 0.0 { achieved_bw / peak } else { 0.0 };
        BandwidthUtilization {
            direction,
            size_bytes,
            peak_bandwidth: peak,
            achieved_bandwidth: achieved_bw,
            utilization_ratio: ratio.clamp(0.0, 1.0),
            transfer_time: observed,
        }
    }
}

/// Bandwidth utilization report for a transfer.
#[derive(Debug, Clone)]
pub struct BandwidthUtilization {
    /// Transfer direction.
    pub direction: CopyDirection,
    /// Bytes transferred.
    pub size_bytes: usize,
    /// Peak theoretical bandwidth in bytes/sec.
    pub peak_bandwidth: f64,
    /// Achieved bandwidth in bytes/sec.
    pub achieved_bandwidth: f64,
    /// Ratio of achieved to peak (0.0–1.0).
    pub utilization_ratio: f64,
    /// Observed transfer time.
    pub transfer_time: Duration,
}

impl fmt::Display for BandwidthUtilization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2} GB/s / {:.2} GB/s ({:.1}%)",
            self.direction,
            self.achieved_bandwidth / 1e9,
            self.peak_bandwidth / 1e9,
            self.utilization_ratio * 100.0,
        )
    }
}

// ── OverlapEstimator ─────────────────────────────────────────────────

/// Estimates overlap between copy and compute operations.
#[derive(Debug, Clone)]
pub struct OverlapEstimator {
    bandwidth_calc: BandwidthCalculator,
}

impl OverlapEstimator {
    /// Create with the given bandwidth calculator.
    pub fn new(bandwidth_calc: BandwidthCalculator) -> Self {
        Self { bandwidth_calc }
    }

    /// Estimate overlap ratio for a copy op running concurrently with compute.
    ///
    /// Returns a value in [0.0, 1.0] where 1.0 means the copy is fully hidden
    /// behind compute and 0.0 means no overlap.
    pub fn estimate_overlap(
        &self,
        copy_op: &AsyncCopyOp,
        compute_duration: Duration,
    ) -> OverlapResult {
        let copy_time = self.bandwidth_calc.estimate_transfer_time(copy_op);
        let copy_secs = copy_time.as_secs_f64();
        let compute_secs = compute_duration.as_secs_f64();

        if copy_secs <= 0.0 || compute_secs <= 0.0 {
            return OverlapResult {
                copy_time,
                compute_time: compute_duration,
                overlap_ratio: 0.0,
                effective_time: copy_time + compute_duration,
                is_fully_hidden: false,
            };
        }

        // The shorter of copy/compute determines the overlap window.
        let overlap = copy_secs.min(compute_secs);
        let overlap_ratio = overlap / copy_secs;
        let effective_secs = copy_secs.max(compute_secs);

        OverlapResult {
            copy_time,
            compute_time: compute_duration,
            overlap_ratio: overlap_ratio.clamp(0.0, 1.0),
            effective_time: Duration::from_secs_f64(effective_secs),
            is_fully_hidden: copy_secs <= compute_secs,
        }
    }

    /// Estimate overlap for a pipeline of copy ops interleaved with compute.
    pub fn estimate_pipeline_overlap(
        &self,
        copy_ops: &[AsyncCopyOp],
        compute_durations: &[Duration],
    ) -> PipelineOverlapResult {
        if copy_ops.is_empty() || compute_durations.is_empty() {
            return PipelineOverlapResult {
                stages: Vec::new(),
                total_sequential_time: Duration::ZERO,
                total_pipelined_time: Duration::ZERO,
                overall_speedup: 1.0,
            };
        }

        let stages: Vec<OverlapResult> = copy_ops
            .iter()
            .zip(compute_durations.iter())
            .map(|(op, dur)| self.estimate_overlap(op, *dur))
            .collect();

        let sequential: f64 =
            stages.iter().map(|s| s.copy_time.as_secs_f64() + s.compute_time.as_secs_f64()).sum();
        let pipelined: f64 = stages.iter().map(|s| s.effective_time.as_secs_f64()).sum();
        let speedup = if pipelined > 0.0 { sequential / pipelined } else { 1.0 };

        PipelineOverlapResult {
            stages,
            total_sequential_time: Duration::from_secs_f64(sequential),
            total_pipelined_time: Duration::from_secs_f64(pipelined),
            overall_speedup: speedup,
        }
    }
}

/// Result of overlap estimation for a single copy/compute pair.
#[derive(Debug, Clone)]
pub struct OverlapResult {
    /// Estimated copy duration.
    pub copy_time: Duration,
    /// Compute duration.
    pub compute_time: Duration,
    /// Fraction of copy time hidden behind compute (0.0–1.0).
    pub overlap_ratio: f64,
    /// Effective wall-clock time (max of copy, compute).
    pub effective_time: Duration,
    /// Whether the copy is fully hidden behind compute.
    pub is_fully_hidden: bool,
}

/// Pipeline overlap estimation across multiple stages.
#[derive(Debug, Clone)]
pub struct PipelineOverlapResult {
    /// Per-stage results.
    pub stages: Vec<OverlapResult>,
    /// Total time if all stages ran sequentially (copy then compute).
    pub total_sequential_time: Duration,
    /// Total time with pipelining.
    pub total_pipelined_time: Duration,
    /// Speedup from pipelining (sequential / pipelined).
    pub overall_speedup: f64,
}

// ── StagedTransferPlan ───────────────────────────────────────────────

/// Strategy for staging transfers through pinned memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingStrategy {
    /// Direct transfer (no staging).
    Direct,
    /// Single-buffered staging through pinned memory.
    SingleBuffered,
    /// Double-buffered: overlap copy-in with copy-out.
    DoubleBuffered,
}

/// A chunk in a staged transfer plan.
#[derive(Debug, Clone)]
pub struct TransferChunk {
    /// Byte offset in the source buffer.
    pub src_offset: usize,
    /// Byte offset in the destination buffer.
    pub dst_offset: usize,
    /// Chunk size in bytes.
    pub size: usize,
    /// Which staging buffer to use (0 or 1 for double-buffering).
    pub buffer_index: usize,
    /// Stream to execute this chunk on.
    pub stream_index: u32,
}

/// A plan for staging a large transfer through pinned memory.
#[derive(Debug, Clone)]
pub struct StagedTransferPlan {
    /// Total bytes to transfer.
    pub total_bytes: usize,
    /// Chunk size for staging.
    pub chunk_size: usize,
    /// Transfer direction.
    pub direction: CopyDirection,
    /// Staging strategy.
    pub strategy: StagingStrategy,
    /// Number of staging buffers required.
    pub num_buffers: usize,
    /// Size of each staging buffer in bytes.
    pub buffer_size: usize,
    /// Ordered list of transfer chunks.
    pub chunks: Vec<TransferChunk>,
}

impl StagedTransferPlan {
    /// Plan a staged transfer.
    pub fn plan(
        total_bytes: usize,
        chunk_size: usize,
        direction: CopyDirection,
        strategy: StagingStrategy,
    ) -> Result<Self> {
        if total_bytes == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "total_bytes must be non-zero".into(),
            }
            .into());
        }
        if chunk_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "chunk_size must be non-zero".into(),
            }
            .into());
        }

        let effective_chunk = chunk_size.min(total_bytes);
        let (num_buffers, buffer_size) = match strategy {
            StagingStrategy::Direct => (0, 0),
            StagingStrategy::SingleBuffered => (1, effective_chunk),
            StagingStrategy::DoubleBuffered => (2, effective_chunk),
        };

        let num_chunks = total_bytes.div_ceil(effective_chunk);
        let mut chunks = Vec::with_capacity(num_chunks);

        for i in 0..num_chunks {
            let offset = i * effective_chunk;
            let size = (total_bytes - offset).min(effective_chunk);
            let buffer_index = match strategy {
                StagingStrategy::Direct => 0,
                StagingStrategy::SingleBuffered => 0,
                StagingStrategy::DoubleBuffered => i % 2,
            };
            // Alternate streams for double-buffered to enable overlap.
            let stream_index = match strategy {
                StagingStrategy::DoubleBuffered => (i % 2) as u32,
                _ => 0,
            };
            chunks.push(TransferChunk {
                src_offset: offset,
                dst_offset: offset,
                size,
                buffer_index,
                stream_index,
            });
        }

        Ok(Self {
            total_bytes,
            chunk_size: effective_chunk,
            direction,
            strategy,
            num_buffers,
            buffer_size,
            chunks,
        })
    }

    /// Total pinned memory required for staging buffers.
    pub fn staging_memory_required(&self) -> usize {
        self.num_buffers * self.buffer_size
    }

    /// Number of transfer chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Estimate total transfer time using the given bandwidth calculator.
    pub fn estimate_time(&self, calc: &BandwidthCalculator) -> Duration {
        let bw = calc.effective_bandwidth(self.direction);
        if bw <= 0.0 {
            return Duration::ZERO;
        }
        match self.strategy {
            StagingStrategy::Direct => {
                let secs = self.total_bytes as f64 / bw;
                calc.base_latency + Duration::from_secs_f64(secs)
            }
            StagingStrategy::SingleBuffered => {
                // Each chunk pays latency + transfer.
                let per_chunk = calc.base_latency.as_secs_f64() + self.chunk_size as f64 / bw;
                Duration::from_secs_f64(per_chunk * self.chunks.len() as f64)
            }
            StagingStrategy::DoubleBuffered => {
                // First chunk is not overlapped; remaining chunks overlap.
                if self.chunks.is_empty() {
                    return Duration::ZERO;
                }
                let chunk_time = calc.base_latency.as_secs_f64() + self.chunk_size as f64 / bw;
                // Pipeline: first chunk + (n-1) * chunk_time (overlapped)
                let total = chunk_time + (self.chunks.len() as f64 - 1.0) * chunk_time;
                Duration::from_secs_f64(total)
            }
        }
    }
}

// ── PinnedMemoryManager ──────────────────────────────────────────────

/// Handle for a pinned memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedAllocId(u64);

impl PinnedAllocId {
    fn next() -> Self {
        Self(next_pinned_id())
    }
}

impl fmt::Display for PinnedAllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pinned-{}", self.0)
    }
}

/// A pinned (page-locked) memory allocation.
#[derive(Debug, Clone)]
pub struct PinnedAllocation {
    /// Unique identifier.
    pub id: PinnedAllocId,
    /// Size in bytes.
    pub size: usize,
    /// Whether currently in use.
    pub in_use: bool,
    /// Label for profiling.
    pub label: String,
}

/// Configuration for the pinned memory manager.
#[derive(Debug, Clone)]
pub struct PinnedMemoryConfig {
    /// Maximum total pinned memory in bytes.
    pub max_pinned_bytes: usize,
    /// Pre-allocate buffers of these sizes.
    pub preallocate_sizes: Vec<usize>,
    /// Alignment for pinned allocations (must be power of two).
    pub alignment: usize,
}

impl Default for PinnedMemoryConfig {
    fn default() -> Self {
        Self {
            max_pinned_bytes: 256 * 1024 * 1024, // 256 MiB
            preallocate_sizes: Vec::new(),
            alignment: 4096, // page-aligned
        }
    }
}

impl PinnedMemoryConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_pinned_bytes == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "max_pinned_bytes must be non-zero".into(),
            }
            .into());
        }
        if !self.alignment.is_power_of_two() {
            return Err(KernelError::InvalidArguments {
                reason: "alignment must be a power of two".into(),
            }
            .into());
        }
        Ok(())
    }
}

/// Manages pinned (page-locked) host memory for efficient DMA transfers.
#[derive(Debug)]
pub struct PinnedMemoryManager {
    config: PinnedMemoryConfig,
    allocations: HashMap<PinnedAllocId, PinnedAllocation>,
    /// Free list: sizes of available pre-allocated buffers.
    free_list: VecDeque<PinnedAllocId>,
    total_allocated: usize,
}

impl PinnedMemoryManager {
    /// Create a new pinned memory manager.
    pub fn new(config: PinnedMemoryConfig) -> Result<Self> {
        config.validate()?;
        let mut mgr = Self {
            config,
            allocations: HashMap::new(),
            free_list: VecDeque::new(),
            total_allocated: 0,
        };
        mgr.preallocate()?;
        Ok(mgr)
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(PinnedMemoryConfig::default())
    }

    fn preallocate(&mut self) -> Result<()> {
        let sizes = self.config.preallocate_sizes.clone();
        for size in &sizes {
            self.allocate_inner(*size, "preallocated", false)?;
        }
        Ok(())
    }

    fn align_up(&self, size: usize) -> usize {
        let mask = self.config.alignment - 1;
        (size + mask) & !mask
    }

    fn allocate_inner(
        &mut self,
        size: usize,
        label: &str,
        mark_in_use: bool,
    ) -> Result<PinnedAllocId> {
        let aligned_size = self.align_up(size);
        if self.total_allocated + aligned_size > self.config.max_pinned_bytes {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "pinned memory limit exceeded: {} + {} > {}",
                    self.total_allocated, aligned_size, self.config.max_pinned_bytes,
                ),
            }
            .into());
        }

        let id = PinnedAllocId::next();
        let alloc =
            PinnedAllocation { id, size: aligned_size, in_use: mark_in_use, label: label.into() };
        self.total_allocated += aligned_size;
        self.allocations.insert(id, alloc);
        if !mark_in_use {
            self.free_list.push_back(id);
        }
        Ok(id)
    }

    /// Allocate pinned memory. Returns a best-fit free block or creates new.
    pub fn allocate(&mut self, size: usize, label: impl Into<String>) -> Result<PinnedAllocId> {
        if size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "allocation size must be non-zero".into(),
            }
            .into());
        }
        let aligned = self.align_up(size);
        let label = label.into();

        // Try to find a free block that fits.
        let candidate = self
            .free_list
            .iter()
            .position(|id| self.allocations.get(id).is_some_and(|a| a.size >= aligned));

        if let Some(idx) = candidate {
            let id = self.free_list.remove(idx).unwrap();
            if let Some(alloc) = self.allocations.get_mut(&id) {
                alloc.in_use = true;
                alloc.label = label;
            }
            return Ok(id);
        }

        self.allocate_inner(size, &label, true)
    }

    /// Free a pinned allocation, returning it to the pool.
    pub fn free(&mut self, id: PinnedAllocId) -> Result<()> {
        let alloc = self.allocations.get_mut(&id).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("unknown pinned allocation: {id}"),
        })?;
        if !alloc.in_use {
            return Err(KernelError::InvalidArguments {
                reason: format!("pinned allocation {id} is already free"),
            }
            .into());
        }
        alloc.in_use = false;
        self.free_list.push_back(id);
        Ok(())
    }

    /// Total pinned memory currently allocated in bytes.
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Total pinned memory currently in use.
    pub fn total_in_use(&self) -> usize {
        self.allocations.values().filter(|a| a.in_use).map(|a| a.size).sum()
    }

    /// Number of live allocations.
    pub fn num_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Number of free blocks available for reuse.
    pub fn num_free(&self) -> usize {
        self.free_list.len()
    }

    /// Get allocation info.
    pub fn get(&self, id: PinnedAllocId) -> Option<&PinnedAllocation> {
        self.allocations.get(&id)
    }

    /// Reset the manager, freeing all allocations.
    pub fn reset(&mut self) {
        self.allocations.clear();
        self.free_list.clear();
        self.total_allocated = 0;
    }
}

// ── CopyScheduler ────────────────────────────────────────────────────

/// Entry in the priority queue (ordered by priority then submission order).
#[derive(Debug, Clone)]
struct SchedulerEntry {
    /// Priority (higher = more urgent).
    priority: u8,
    /// Submission order (lower = earlier).
    sequence: u64,
    /// The copy operation.
    op: AsyncCopyOp,
}

impl PartialEq for SchedulerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for SchedulerEntry {}

impl PartialOrd for SchedulerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedulerEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority).then_with(|| other.sequence.cmp(&self.sequence))
    }
}

/// Scheduling policy for copy operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePolicy {
    /// First-in-first-out (FIFO), ignoring priorities.
    Fifo,
    /// Priority-based with FIFO tie-breaking.
    PriorityFifo,
    /// Coalesce small transfers into larger batches when possible.
    Coalescing { max_batch_bytes: usize },
}

/// Result of scheduling: a batch of ops to execute.
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    /// Operations in execution order.
    pub ops: Vec<AsyncCopyOp>,
    /// Total bytes in this batch.
    pub total_bytes: usize,
}

/// Schedules async copy operations with priority and dependency tracking.
#[derive(Debug)]
pub struct CopyScheduler {
    policy: SchedulePolicy,
    queue: BinaryHeap<SchedulerEntry>,
    completed: HashMap<CopyId, AsyncCopyOp>,
    next_sequence: u64,
}

impl CopyScheduler {
    /// Create a new scheduler with the given policy.
    pub fn new(policy: SchedulePolicy) -> Self {
        Self { policy, queue: BinaryHeap::new(), completed: HashMap::new(), next_sequence: 0 }
    }

    /// Submit a copy operation to the scheduler.
    pub fn submit(&mut self, op: AsyncCopyOp) -> CopyId {
        let id = op.id;
        let priority = match self.policy {
            SchedulePolicy::Fifo => 0, // all same priority for FIFO
            _ => op.priority.as_numeric(),
        };
        let entry = SchedulerEntry { priority, sequence: self.next_sequence, op };
        self.next_sequence += 1;
        self.queue.push(entry);
        id
    }

    /// Mark an operation as completed.
    pub fn mark_completed(&mut self, mut op: AsyncCopyOp) {
        op.status = CopyStatus::Completed;
        self.completed.insert(op.id, op);
    }

    /// Check whether all dependencies of an op are satisfied.
    fn deps_satisfied(&self, op: &AsyncCopyOp) -> bool {
        op.depends_on.iter().all(|dep| self.completed.contains_key(dep))
    }

    /// Dequeue the next batch of ops whose dependencies are satisfied.
    pub fn next_batch(&mut self) -> ScheduledBatch {
        let mut ready = Vec::new();
        let mut deferred = Vec::new();
        let mut total_bytes = 0usize;

        while let Some(entry) = self.queue.pop() {
            if self.deps_satisfied(&entry.op) {
                let should_add = match self.policy {
                    SchedulePolicy::Coalescing { max_batch_bytes } => {
                        total_bytes + entry.op.size_bytes <= max_batch_bytes || ready.is_empty()
                    }
                    _ => true,
                };
                if should_add {
                    total_bytes += entry.op.size_bytes;
                    let mut op = entry.op;
                    op.status = CopyStatus::InProgress;
                    ready.push(op);
                } else {
                    deferred.push(entry);
                }
            } else {
                deferred.push(entry);
            }
        }

        // Put deferred items back.
        for entry in deferred {
            self.queue.push(entry);
        }

        ScheduledBatch { ops: ready, total_bytes }
    }

    /// Number of pending operations in the queue.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Number of completed operations.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clear all state.
    pub fn reset(&mut self) {
        self.queue.clear();
        self.completed.clear();
        self.next_sequence = 0;
    }
}

// ── Simulate async copy execution (CPU fallback) ─────────────────────

/// Execute a batch of copy operations (CPU simulation).
pub fn execute_copy_batch(
    batch: &ScheduledBatch,
    calc: &BandwidthCalculator,
) -> Vec<(CopyId, Duration)> {
    batch
        .ops
        .iter()
        .map(|op| {
            let dur = calc.estimate_transfer_time(op);
            (op.id, dur)
        })
        .collect()
}

/// Plan and estimate a complete transfer pipeline.
pub fn plan_transfer_pipeline(
    total_bytes: usize,
    chunk_size: usize,
    direction: CopyDirection,
    strategy: StagingStrategy,
    calc: &BandwidthCalculator,
) -> Result<(StagedTransferPlan, Duration)> {
    let plan = StagedTransferPlan::plan(total_bytes, chunk_size, direction, strategy)?;
    let time = plan.estimate_time(calc);
    Ok((plan, time))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- CopyDirection tests --

    #[test]
    fn copy_direction_display_h2d() {
        assert_eq!(CopyDirection::HostToDevice.to_string(), "H2D");
    }

    #[test]
    fn copy_direction_display_d2h() {
        assert_eq!(CopyDirection::DeviceToHost.to_string(), "D2H");
    }

    #[test]
    fn copy_direction_display_d2d() {
        assert_eq!(CopyDirection::DeviceToDevice.to_string(), "D2D");
    }

    #[test]
    fn copy_direction_display_p2p() {
        let d = CopyDirection::PeerToPeer { src_device: 0, dst_device: 1 };
        assert_eq!(d.to_string(), "P2P(0->1)");
    }

    // -- CopyPriority tests --

    #[test]
    fn copy_priority_default_is_normal() {
        assert_eq!(CopyPriority::default(), CopyPriority::Normal);
    }

    #[test]
    fn copy_priority_ordering() {
        assert!(CopyPriority::Background < CopyPriority::Normal);
        assert!(CopyPriority::Normal < CopyPriority::High);
        assert!(CopyPriority::High < CopyPriority::Critical);
    }

    #[test]
    fn copy_priority_numeric() {
        assert_eq!(CopyPriority::Background.as_numeric(), 0);
        assert_eq!(CopyPriority::Normal.as_numeric(), 1);
        assert_eq!(CopyPriority::High.as_numeric(), 2);
        assert_eq!(CopyPriority::Critical.as_numeric(), 3);
    }

    // -- AsyncCopyOp tests --

    #[test]
    fn async_copy_op_new_defaults() {
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 1024, "test");
        assert_eq!(op.direction, CopyDirection::HostToDevice);
        assert_eq!(op.size_bytes, 1024);
        assert_eq!(op.priority, CopyPriority::Normal);
        assert_eq!(op.status, CopyStatus::Pending);
        assert!(!op.uses_pinned_memory);
        assert!(op.depends_on.is_empty());
        assert!(op.estimated_duration.is_none());
    }

    #[test]
    fn async_copy_op_builder_chain() {
        let other_id = CopyId::next();
        let op = AsyncCopyOp::new(CopyDirection::DeviceToHost, 2048, "weights")
            .with_priority(CopyPriority::High)
            .with_stream(2)
            .with_pinned_memory()
            .depends_on(other_id);
        assert_eq!(op.priority, CopyPriority::High);
        assert_eq!(op.stream_index, 2);
        assert!(op.uses_pinned_memory);
        assert_eq!(op.depends_on.len(), 1);
    }

    #[test]
    fn copy_id_unique() {
        let a = CopyId::next();
        let b = CopyId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn copy_id_display() {
        let id = CopyId(42);
        assert_eq!(id.to_string(), "copy-42");
    }

    // -- InterconnectType tests --

    #[test]
    fn interconnect_pcie_gen3_bandwidth() {
        let bw = InterconnectType::PcieGen3.peak_bandwidth_bytes_per_sec();
        assert!((bw - 15.75e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_pcie_gen4_bandwidth() {
        let bw = InterconnectType::PcieGen4.peak_bandwidth_bytes_per_sec();
        assert!((bw - 31.5e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_pcie_gen5_bandwidth() {
        let bw = InterconnectType::PcieGen5.peak_bandwidth_bytes_per_sec();
        assert!((bw - 63.0e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_nvlink2_bandwidth() {
        let bw = InterconnectType::NvLink2.peak_bandwidth_bytes_per_sec();
        assert!((bw - 150.0e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_nvlink3_bandwidth() {
        let bw = InterconnectType::NvLink3.peak_bandwidth_bytes_per_sec();
        assert!((bw - 300.0e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_nvlink4_bandwidth() {
        let bw = InterconnectType::NvLink4.peak_bandwidth_bytes_per_sec();
        assert!((bw - 450.0e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_custom_bandwidth() {
        let bw = InterconnectType::Custom { bandwidth_gbps: 100.0 }.peak_bandwidth_bytes_per_sec();
        assert!((bw - 100.0e9).abs() < 1e6);
    }

    #[test]
    fn interconnect_p2p_support() {
        assert!(!InterconnectType::PcieGen3.supports_p2p());
        assert!(!InterconnectType::PcieGen4.supports_p2p());
        assert!(!InterconnectType::PcieGen5.supports_p2p());
        assert!(InterconnectType::NvLink2.supports_p2p());
        assert!(InterconnectType::NvLink3.supports_p2p());
        assert!(InterconnectType::NvLink4.supports_p2p());
        assert!(!InterconnectType::Custom { bandwidth_gbps: 50.0 }.supports_p2p());
    }

    // -- BandwidthCalculator tests --

    #[test]
    fn bandwidth_calc_default() {
        let calc = BandwidthCalculator::default();
        assert_eq!(calc.host_device_interconnect, InterconnectType::PcieGen4);
        assert!((calc.efficiency_factor - 0.85).abs() < 1e-9);
        assert_eq!(calc.overhead_bytes, 64);
    }

    #[test]
    fn bandwidth_calc_effective_h2d() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4).with_efficiency(1.0);
        let bw = calc.effective_bandwidth(CopyDirection::HostToDevice);
        assert!((bw - 31.5e9).abs() < 1e6);
    }

    #[test]
    fn bandwidth_calc_effective_d2d_double() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4).with_efficiency(1.0);
        let bw = calc.effective_bandwidth(CopyDirection::DeviceToDevice);
        assert!((bw - 63.0e9).abs() < 1e6); // 2× PCIe Gen4
    }

    #[test]
    fn bandwidth_calc_effective_p2p_uses_p2p_interconnect() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_p2p_interconnect(InterconnectType::NvLink3);
        let bw =
            calc.effective_bandwidth(CopyDirection::PeerToPeer { src_device: 0, dst_device: 1 });
        assert!((bw - 300.0e9).abs() < 1e6);
    }

    #[test]
    fn bandwidth_calc_p2p_falls_back_to_host_device() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4).with_efficiency(1.0);
        let bw =
            calc.effective_bandwidth(CopyDirection::PeerToPeer { src_device: 0, dst_device: 1 });
        assert!((bw - 31.5e9).abs() < 1e6); // falls back to PCIe
    }

    #[test]
    fn bandwidth_calc_efficiency_clamped() {
        let calc = BandwidthCalculator::default().with_efficiency(1.5);
        assert!((calc.efficiency_factor - 1.0).abs() < 1e-9);
        let calc2 = BandwidthCalculator::default().with_efficiency(-0.3);
        assert!(calc2.efficiency_factor.abs() < 1e-9);
    }

    #[test]
    fn bandwidth_calc_estimate_small_transfer() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(Duration::ZERO);
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 31_500_000_000, "big");
        let dur = calc.estimate_transfer_time(&op);
        // ~1 sec plus overhead for ~31.5 GB at 31.5 GB/s
        assert!(dur.as_secs_f64() > 0.99 && dur.as_secs_f64() < 1.01);
    }

    #[test]
    fn bandwidth_calc_estimate_includes_latency() {
        let latency = Duration::from_millis(10);
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(latency);
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 0, "tiny");
        // Even zero bytes should include base latency + overhead.
        let dur = calc.estimate_transfer_time(&op);
        assert!(dur >= latency);
    }

    #[test]
    fn bandwidth_utilization_calculation() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4);
        let util =
            calc.utilization(CopyDirection::HostToDevice, 31_500_000_000, Duration::from_secs(2));
        assert_eq!(util.size_bytes, 31_500_000_000);
        // Achieved ≈ 15.75 GB/s, peak ≈ 31.5 GB/s → utilization ≈ 50%
        assert!((util.utilization_ratio - 0.5).abs() < 0.01);
    }

    #[test]
    fn bandwidth_utilization_zero_time() {
        let calc = BandwidthCalculator::default();
        let util = calc.utilization(CopyDirection::HostToDevice, 1024, Duration::ZERO);
        assert!(util.achieved_bandwidth.abs() < 1e-9);
    }

    #[test]
    fn bandwidth_utilization_display() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4);
        let util =
            calc.utilization(CopyDirection::HostToDevice, 31_500_000_000, Duration::from_secs(1));
        let s = util.to_string();
        assert!(s.contains("H2D"));
        assert!(s.contains("GB/s"));
    }

    // -- OverlapEstimator tests --

    #[test]
    fn overlap_fully_hidden() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(Duration::ZERO);
        let est = OverlapEstimator::new(calc);
        // Small copy, long compute → fully hidden.
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 1_000_000, "small");
        let result = est.estimate_overlap(&op, Duration::from_secs(10));
        assert!(result.is_fully_hidden);
        assert!((result.overlap_ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn overlap_not_hidden() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen3)
            .with_efficiency(1.0)
            .with_base_latency(Duration::ZERO);
        let est = OverlapEstimator::new(calc);
        // Large copy, short compute → not fully hidden.
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 15_750_000_000, "big");
        let result = est.estimate_overlap(&op, Duration::from_millis(100));
        assert!(!result.is_fully_hidden);
        assert!(result.overlap_ratio < 0.2);
    }

    #[test]
    fn overlap_zero_compute() {
        let calc = BandwidthCalculator::default();
        let est = OverlapEstimator::new(calc);
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 1024, "x");
        let result = est.estimate_overlap(&op, Duration::ZERO);
        assert!(!result.is_fully_hidden);
        assert!(result.overlap_ratio.abs() < 1e-9);
    }

    #[test]
    fn overlap_zero_copy_size_with_overhead() {
        let calc = BandwidthCalculator::default().with_base_latency(Duration::ZERO);
        let est = OverlapEstimator::new(calc);
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 0, "empty");
        // Even empty copy has overhead bytes.
        let result = est.estimate_overlap(&op, Duration::from_secs(1));
        assert!(result.is_fully_hidden);
    }

    #[test]
    fn pipeline_overlap_empty() {
        let calc = BandwidthCalculator::default();
        let est = OverlapEstimator::new(calc);
        let result = est.estimate_pipeline_overlap(&[], &[]);
        assert!(result.stages.is_empty());
        assert!((result.overall_speedup - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pipeline_overlap_multiple_stages() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(Duration::ZERO);
        let est = OverlapEstimator::new(calc);
        let ops = vec![
            AsyncCopyOp::new(CopyDirection::HostToDevice, 1_000_000, "chunk1"),
            AsyncCopyOp::new(CopyDirection::HostToDevice, 1_000_000, "chunk2"),
        ];
        let computes = vec![Duration::from_secs(1), Duration::from_secs(1)];
        let result = est.estimate_pipeline_overlap(&ops, &computes);
        assert_eq!(result.stages.len(), 2);
        assert!(result.overall_speedup > 1.0);
    }

    #[test]
    fn pipeline_overlap_mismatched_lengths() {
        let calc = BandwidthCalculator::default();
        let est = OverlapEstimator::new(calc);
        let ops = vec![
            AsyncCopyOp::new(CopyDirection::HostToDevice, 1024, "a"),
            AsyncCopyOp::new(CopyDirection::HostToDevice, 2048, "b"),
            AsyncCopyOp::new(CopyDirection::HostToDevice, 4096, "c"),
        ];
        let computes = vec![Duration::from_millis(10)]; // only 1
        let result = est.estimate_pipeline_overlap(&ops, &computes);
        assert_eq!(result.stages.len(), 1); // zip truncates
    }

    // -- StagedTransferPlan tests --

    #[test]
    fn staged_plan_direct() {
        let plan = StagedTransferPlan::plan(
            1024,
            256,
            CopyDirection::HostToDevice,
            StagingStrategy::Direct,
        )
        .unwrap();
        assert_eq!(plan.num_buffers, 0);
        assert_eq!(plan.staging_memory_required(), 0);
        assert_eq!(plan.num_chunks(), 4);
    }

    #[test]
    fn staged_plan_single_buffered() {
        let plan = StagedTransferPlan::plan(
            1000,
            256,
            CopyDirection::HostToDevice,
            StagingStrategy::SingleBuffered,
        )
        .unwrap();
        assert_eq!(plan.num_buffers, 1);
        assert_eq!(plan.buffer_size, 256);
        assert_eq!(plan.staging_memory_required(), 256);
        assert_eq!(plan.num_chunks(), 4); // 256+256+256+232
        // All chunks use buffer 0.
        for chunk in &plan.chunks {
            assert_eq!(chunk.buffer_index, 0);
            assert_eq!(chunk.stream_index, 0);
        }
    }

    #[test]
    fn staged_plan_double_buffered() {
        let plan = StagedTransferPlan::plan(
            2048,
            512,
            CopyDirection::DeviceToHost,
            StagingStrategy::DoubleBuffered,
        )
        .unwrap();
        assert_eq!(plan.num_buffers, 2);
        assert_eq!(plan.staging_memory_required(), 1024);
        assert_eq!(plan.num_chunks(), 4);
        // Alternating buffer indices.
        assert_eq!(plan.chunks[0].buffer_index, 0);
        assert_eq!(plan.chunks[1].buffer_index, 1);
        assert_eq!(plan.chunks[2].buffer_index, 0);
        assert_eq!(plan.chunks[3].buffer_index, 1);
        // Alternating stream indices.
        assert_eq!(plan.chunks[0].stream_index, 0);
        assert_eq!(plan.chunks[1].stream_index, 1);
        assert_eq!(plan.chunks[2].stream_index, 0);
        assert_eq!(plan.chunks[3].stream_index, 1);
    }

    #[test]
    fn staged_plan_chunk_larger_than_total() {
        let plan = StagedTransferPlan::plan(
            100,
            1024,
            CopyDirection::HostToDevice,
            StagingStrategy::SingleBuffered,
        )
        .unwrap();
        assert_eq!(plan.chunk_size, 100); // clamped
        assert_eq!(plan.num_chunks(), 1);
        assert_eq!(plan.chunks[0].size, 100);
    }

    #[test]
    fn staged_plan_exact_multiple() {
        let plan = StagedTransferPlan::plan(
            1024,
            512,
            CopyDirection::HostToDevice,
            StagingStrategy::Direct,
        )
        .unwrap();
        assert_eq!(plan.num_chunks(), 2);
        assert_eq!(plan.chunks[0].size, 512);
        assert_eq!(plan.chunks[1].size, 512);
    }

    #[test]
    fn staged_plan_non_exact_multiple() {
        let plan = StagedTransferPlan::plan(
            1000,
            300,
            CopyDirection::HostToDevice,
            StagingStrategy::Direct,
        )
        .unwrap();
        assert_eq!(plan.num_chunks(), 4); // 300+300+300+100
        assert_eq!(plan.chunks[3].size, 100);
    }

    #[test]
    fn staged_plan_zero_bytes_error() {
        let result =
            StagedTransferPlan::plan(0, 256, CopyDirection::HostToDevice, StagingStrategy::Direct);
        assert!(result.is_err());
    }

    #[test]
    fn staged_plan_zero_chunk_error() {
        let result =
            StagedTransferPlan::plan(1024, 0, CopyDirection::HostToDevice, StagingStrategy::Direct);
        assert!(result.is_err());
    }

    #[test]
    fn staged_plan_offsets_contiguous() {
        let plan = StagedTransferPlan::plan(
            1000,
            256,
            CopyDirection::HostToDevice,
            StagingStrategy::Direct,
        )
        .unwrap();
        let mut expected_offset = 0;
        for chunk in &plan.chunks {
            assert_eq!(chunk.src_offset, expected_offset);
            assert_eq!(chunk.dst_offset, expected_offset);
            expected_offset += chunk.size;
        }
        assert_eq!(expected_offset, 1000);
    }

    #[test]
    fn staged_plan_estimate_time_direct() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(Duration::ZERO);
        let plan = StagedTransferPlan::plan(
            31_500_000_000,
            1_000_000_000,
            CopyDirection::HostToDevice,
            StagingStrategy::Direct,
        )
        .unwrap();
        let dur = plan.estimate_time(&calc);
        // ~1 second for 31.5 GB at 31.5 GB/s
        assert!(dur.as_secs_f64() > 0.99 && dur.as_secs_f64() < 1.01);
    }

    #[test]
    fn staged_plan_estimate_time_single_buffered() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(Duration::from_micros(1));
        let plan = StagedTransferPlan::plan(
            1024,
            512,
            CopyDirection::HostToDevice,
            StagingStrategy::SingleBuffered,
        )
        .unwrap();
        let dur = plan.estimate_time(&calc);
        assert!(dur > Duration::ZERO);
    }

    #[test]
    fn staged_plan_estimate_double_buffered_no_faster_than_single_chunk() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(1.0)
            .with_base_latency(Duration::ZERO);
        // 1 chunk: both strategies should be roughly the same.
        let plan = StagedTransferPlan::plan(
            512,
            1024,
            CopyDirection::HostToDevice,
            StagingStrategy::DoubleBuffered,
        )
        .unwrap();
        let dur = plan.estimate_time(&calc);
        assert!(dur > Duration::ZERO);
        assert_eq!(plan.num_chunks(), 1);
    }

    // -- PinnedMemoryManager tests --

    #[test]
    fn pinned_mgr_default() {
        let mgr = PinnedMemoryManager::with_defaults().unwrap();
        assert_eq!(mgr.total_allocated(), 0);
        assert_eq!(mgr.num_allocations(), 0);
    }

    #[test]
    fn pinned_mgr_allocate_and_free() {
        let mut mgr = PinnedMemoryManager::with_defaults().unwrap();
        let id = mgr.allocate(1024, "test").unwrap();
        assert!(mgr.get(id).unwrap().in_use);
        assert!(mgr.total_in_use() > 0);

        mgr.free(id).unwrap();
        assert!(!mgr.get(id).unwrap().in_use);
        assert_eq!(mgr.total_in_use(), 0);
    }

    #[test]
    fn pinned_mgr_reuse_freed_block() {
        let mut mgr = PinnedMemoryManager::with_defaults().unwrap();
        let id1 = mgr.allocate(4096, "first").unwrap();
        mgr.free(id1).unwrap();
        let allocated_after_free = mgr.total_allocated();

        // Should reuse the freed block.
        let id2 = mgr.allocate(2048, "second").unwrap();
        assert_eq!(mgr.total_allocated(), allocated_after_free); // no new allocation
        assert_eq!(id2, id1); // same block reused
    }

    #[test]
    fn pinned_mgr_alignment() {
        let config = PinnedMemoryConfig {
            max_pinned_bytes: 1024 * 1024,
            preallocate_sizes: Vec::new(),
            alignment: 4096,
        };
        let mut mgr = PinnedMemoryManager::new(config).unwrap();
        let id = mgr.allocate(100, "small").unwrap();
        assert_eq!(mgr.get(id).unwrap().size, 4096); // aligned up
    }

    #[test]
    fn pinned_mgr_limit_exceeded() {
        let config = PinnedMemoryConfig {
            max_pinned_bytes: 8192,
            preallocate_sizes: Vec::new(),
            alignment: 4096,
        };
        let mut mgr = PinnedMemoryManager::new(config).unwrap();
        let _id1 = mgr.allocate(4096, "a").unwrap();
        let _id2 = mgr.allocate(4096, "b").unwrap();
        let result = mgr.allocate(4096, "c"); // exceeds limit
        assert!(result.is_err());
    }

    #[test]
    fn pinned_mgr_double_free_error() {
        let mut mgr = PinnedMemoryManager::with_defaults().unwrap();
        let id = mgr.allocate(1024, "x").unwrap();
        mgr.free(id).unwrap();
        assert!(mgr.free(id).is_err()); // already free
    }

    #[test]
    fn pinned_mgr_free_unknown_error() {
        let mut mgr = PinnedMemoryManager::with_defaults().unwrap();
        let bogus = PinnedAllocId(99999);
        assert!(mgr.free(bogus).is_err());
    }

    #[test]
    fn pinned_mgr_zero_size_error() {
        let mut mgr = PinnedMemoryManager::with_defaults().unwrap();
        assert!(mgr.allocate(0, "zero").is_err());
    }

    #[test]
    fn pinned_mgr_preallocate() {
        let config = PinnedMemoryConfig {
            max_pinned_bytes: 1024 * 1024,
            preallocate_sizes: vec![4096, 8192],
            alignment: 4096,
        };
        let mgr = PinnedMemoryManager::new(config).unwrap();
        assert_eq!(mgr.num_allocations(), 2);
        assert_eq!(mgr.num_free(), 2); // preallocated → free list
        assert_eq!(mgr.total_in_use(), 0);
    }

    #[test]
    fn pinned_mgr_reset() {
        let mut mgr = PinnedMemoryManager::with_defaults().unwrap();
        let _id = mgr.allocate(4096, "a").unwrap();
        mgr.reset();
        assert_eq!(mgr.total_allocated(), 0);
        assert_eq!(mgr.num_allocations(), 0);
        assert_eq!(mgr.num_free(), 0);
    }

    #[test]
    fn pinned_config_validate_zero_max() {
        let config = PinnedMemoryConfig { max_pinned_bytes: 0, ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pinned_config_validate_bad_alignment() {
        let config = PinnedMemoryConfig {
            alignment: 3, // not power of two
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn pinned_alloc_id_display() {
        let id = PinnedAllocId(7);
        assert_eq!(id.to_string(), "pinned-7");
    }

    // -- CopyScheduler tests --

    #[test]
    fn scheduler_fifo_ordering() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        let op1 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "first")
            .with_priority(CopyPriority::Background);
        let op2 = AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "second")
            .with_priority(CopyPriority::Critical);
        sched.submit(op1);
        sched.submit(op2);
        let batch = sched.next_batch();
        // FIFO: both dequeued, first submitted comes first.
        assert_eq!(batch.ops.len(), 2);
        assert_eq!(batch.ops[0].label, "first");
        assert_eq!(batch.ops[1].label, "second");
    }

    #[test]
    fn scheduler_priority_ordering() {
        let mut sched = CopyScheduler::new(SchedulePolicy::PriorityFifo);
        let op_low = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "low")
            .with_priority(CopyPriority::Background);
        let op_high = AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "high")
            .with_priority(CopyPriority::Critical);
        sched.submit(op_low);
        sched.submit(op_high);
        let batch = sched.next_batch();
        assert_eq!(batch.ops[0].label, "high");
        assert_eq!(batch.ops[1].label, "low");
    }

    #[test]
    fn scheduler_priority_fifo_tiebreak() {
        let mut sched = CopyScheduler::new(SchedulePolicy::PriorityFifo);
        let op1 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "first")
            .with_priority(CopyPriority::Normal);
        let op2 = AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "second")
            .with_priority(CopyPriority::Normal);
        sched.submit(op1);
        sched.submit(op2);
        let batch = sched.next_batch();
        assert_eq!(batch.ops[0].label, "first");
        assert_eq!(batch.ops[1].label, "second");
    }

    #[test]
    fn scheduler_dependency_blocking() {
        let mut sched = CopyScheduler::new(SchedulePolicy::PriorityFifo);
        let op1 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "first");
        let id1 = op1.id;
        let op2 = AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "second").depends_on(id1);
        sched.submit(op1);
        sched.submit(op2);

        // First batch: only op1 (op2 blocked by dep).
        let batch1 = sched.next_batch();
        assert_eq!(batch1.ops.len(), 1);
        assert_eq!(batch1.ops[0].label, "first");

        // Mark op1 complete.
        sched.mark_completed(batch1.ops[0].clone());

        // Second batch: op2 now ready.
        let batch2 = sched.next_batch();
        assert_eq!(batch2.ops.len(), 1);
        assert_eq!(batch2.ops[0].label, "second");
    }

    #[test]
    fn scheduler_coalescing_respects_limit() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Coalescing { max_batch_bytes: 500 });
        sched.submit(AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "a"));
        sched.submit(AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "b"));
        sched.submit(AsyncCopyOp::new(CopyDirection::HostToDevice, 200, "c"));

        let batch = sched.next_batch();
        // 200+200 = 400 ≤ 500, but 400+200 = 600 > 500 → only 2.
        assert_eq!(batch.ops.len(), 2);
        assert_eq!(batch.total_bytes, 400);

        // Remaining 1 op.
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn scheduler_coalescing_always_takes_one() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Coalescing { max_batch_bytes: 100 });
        sched.submit(AsyncCopyOp::new(CopyDirection::HostToDevice, 500, "big"));
        let batch = sched.next_batch();
        // Always takes at least one even if it exceeds the limit.
        assert_eq!(batch.ops.len(), 1);
    }

    #[test]
    fn scheduler_empty_batch() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        let batch = sched.next_batch();
        assert!(batch.ops.is_empty());
        assert_eq!(batch.total_bytes, 0);
    }

    #[test]
    fn scheduler_pending_and_completed_counts() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        assert_eq!(sched.pending_count(), 0);
        assert_eq!(sched.completed_count(), 0);
        assert!(sched.is_empty());

        sched.submit(AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "a"));
        assert_eq!(sched.pending_count(), 1);
        assert!(!sched.is_empty());

        let batch = sched.next_batch();
        sched.mark_completed(batch.ops[0].clone());
        assert_eq!(sched.completed_count(), 1);
        assert!(sched.is_empty());
    }

    #[test]
    fn scheduler_reset() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        sched.submit(AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "a"));
        sched.reset();
        assert!(sched.is_empty());
        assert_eq!(sched.completed_count(), 0);
    }

    #[test]
    fn scheduler_multiple_deps() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        let op1 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "dep1");
        let op2 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "dep2");
        let id1 = op1.id;
        let id2 = op2.id;
        let op3 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "waiter")
            .depends_on(id1)
            .depends_on(id2);

        sched.submit(op1);
        sched.submit(op2);
        sched.submit(op3);

        // First batch: op1 and op2 (no deps).
        let batch1 = sched.next_batch();
        assert_eq!(batch1.ops.len(), 2);

        // op3 still blocked.
        let batch_empty = sched.next_batch();
        assert!(batch_empty.ops.is_empty());

        // Complete both deps.
        for op in batch1.ops {
            sched.mark_completed(op);
        }

        // Now op3 is ready.
        let batch2 = sched.next_batch();
        assert_eq!(batch2.ops.len(), 1);
        assert_eq!(batch2.ops[0].label, "waiter");
    }

    #[test]
    fn scheduler_status_transitions() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        let op = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "a");
        assert_eq!(op.status, CopyStatus::Pending);

        sched.submit(op);
        let batch = sched.next_batch();
        assert_eq!(batch.ops[0].status, CopyStatus::InProgress);

        let mut completed = batch.ops[0].clone();
        sched.mark_completed(completed.clone());
        completed = sched.completed.get(&completed.id).cloned().unwrap();
        assert_eq!(completed.status, CopyStatus::Completed);
    }

    // -- execute_copy_batch tests --

    #[test]
    fn execute_batch_returns_durations() {
        let calc = BandwidthCalculator::default();
        let op1 = AsyncCopyOp::new(CopyDirection::HostToDevice, 1024, "a");
        let op2 = AsyncCopyOp::new(CopyDirection::DeviceToHost, 2048, "b");
        let batch = ScheduledBatch { ops: vec![op1, op2], total_bytes: 3072 };
        let results = execute_copy_batch(&batch, &calc);
        assert_eq!(results.len(), 2);
        for (_, dur) in &results {
            assert!(*dur > Duration::ZERO);
        }
    }

    #[test]
    fn execute_empty_batch() {
        let calc = BandwidthCalculator::default();
        let batch = ScheduledBatch { ops: vec![], total_bytes: 0 };
        let results = execute_copy_batch(&batch, &calc);
        assert!(results.is_empty());
    }

    // -- plan_transfer_pipeline tests --

    #[test]
    fn plan_pipeline_success() {
        let calc = BandwidthCalculator::default();
        let (plan, time) = plan_transfer_pipeline(
            10_000,
            1_000,
            CopyDirection::HostToDevice,
            StagingStrategy::DoubleBuffered,
            &calc,
        )
        .unwrap();
        assert_eq!(plan.num_chunks(), 10);
        assert!(time > Duration::ZERO);
    }

    #[test]
    fn plan_pipeline_error_zero_bytes() {
        let calc = BandwidthCalculator::default();
        let result = plan_transfer_pipeline(
            0,
            1024,
            CopyDirection::HostToDevice,
            StagingStrategy::Direct,
            &calc,
        );
        assert!(result.is_err());
    }

    // -- Edge cases and integration --

    #[test]
    fn end_to_end_staged_double_buffer_with_scheduler() {
        let calc = BandwidthCalculator::new(InterconnectType::PcieGen4)
            .with_efficiency(0.9)
            .with_base_latency(Duration::from_micros(2));

        let plan = StagedTransferPlan::plan(
            8192,
            2048,
            CopyDirection::HostToDevice,
            StagingStrategy::DoubleBuffered,
        )
        .unwrap();
        assert_eq!(plan.num_chunks(), 4);

        let mut sched = CopyScheduler::new(SchedulePolicy::PriorityFifo);
        let mut prev_id = None;

        for chunk in &plan.chunks {
            let mut op = AsyncCopyOp::new(
                CopyDirection::HostToDevice,
                chunk.size,
                format!("chunk@{}", chunk.src_offset),
            )
            .with_stream(chunk.stream_index);
            if let Some(dep) = prev_id {
                op = op.depends_on(dep);
            }
            let id = op.id;
            sched.submit(op);
            prev_id = Some(id);
        }

        // Execute chunk by chunk.
        let mut executed = 0;
        while !sched.is_empty() {
            let batch = sched.next_batch();
            if batch.ops.is_empty() {
                break;
            }
            for op in &batch.ops {
                let dur = calc.estimate_transfer_time(op);
                assert!(dur > Duration::ZERO);
                sched.mark_completed(op.clone());
                executed += 1;
            }
        }
        assert_eq!(executed, 4);
    }

    #[test]
    fn end_to_end_bandwidth_and_overlap() {
        let calc = BandwidthCalculator::new(InterconnectType::NvLink3)
            .with_efficiency(0.8)
            .with_base_latency(Duration::from_micros(1));
        let est = OverlapEstimator::new(calc.clone());

        // 1 GB transfer over NVLink3 at 80% efficiency.
        let op = AsyncCopyOp::new(
            CopyDirection::PeerToPeer { src_device: 0, dst_device: 1 },
            1_000_000_000,
            "p2p-weights",
        );
        let transfer_time = calc.estimate_transfer_time(&op);
        // ~1GB / (300 * 0.8) GB/s ≈ 4.17ms
        assert!(transfer_time.as_secs_f64() < 0.01);

        let result = est.estimate_overlap(&op, Duration::from_millis(50));
        assert!(result.is_fully_hidden);

        let util = calc.utilization(op.direction, op.size_bytes, transfer_time);
        // Should be close to 80% efficiency.
        assert!(util.utilization_ratio > 0.7 && util.utilization_ratio < 0.9);
    }

    #[test]
    fn end_to_end_pinned_memory_with_staging() {
        let config = PinnedMemoryConfig {
            max_pinned_bytes: 64 * 1024,
            preallocate_sizes: Vec::new(),
            alignment: 4096,
        };
        let mut pmm = PinnedMemoryManager::new(config).unwrap();

        let plan = StagedTransferPlan::plan(
            32768,
            8192,
            CopyDirection::HostToDevice,
            StagingStrategy::DoubleBuffered,
        )
        .unwrap();

        // Allocate staging buffers.
        let mut buf_ids = Vec::new();
        for i in 0..plan.num_buffers {
            let id = pmm.allocate(plan.buffer_size, format!("staging-{i}")).unwrap();
            buf_ids.push(id);
        }
        assert_eq!(buf_ids.len(), 2);
        assert_eq!(pmm.total_in_use(), 2 * 8192);

        // Free staging buffers.
        for id in buf_ids {
            pmm.free(id).unwrap();
        }
        assert_eq!(pmm.total_in_use(), 0);
    }

    #[test]
    fn copy_status_values_distinct() {
        let statuses = [
            CopyStatus::Pending,
            CopyStatus::InProgress,
            CopyStatus::Completed,
            CopyStatus::Failed,
            CopyStatus::Cancelled,
        ];
        for (i, a) in statuses.iter().enumerate() {
            for (j, b) in statuses.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn staging_strategy_equality() {
        assert_eq!(StagingStrategy::Direct, StagingStrategy::Direct);
        assert_ne!(StagingStrategy::Direct, StagingStrategy::SingleBuffered);
        assert_ne!(StagingStrategy::SingleBuffered, StagingStrategy::DoubleBuffered);
    }

    #[test]
    fn bandwidth_calc_with_base_latency() {
        let latency = Duration::from_millis(5);
        let calc = BandwidthCalculator::default().with_base_latency(latency);
        assert_eq!(calc.base_latency, latency);
    }

    #[test]
    fn interconnect_custom_zero_bandwidth() {
        let bw = InterconnectType::Custom { bandwidth_gbps: 0.0 }.peak_bandwidth_bytes_per_sec();
        assert!(bw.abs() < 1e-9);
    }

    #[test]
    fn async_copy_op_label_preserved() {
        let op = AsyncCopyOp::new(CopyDirection::DeviceToDevice, 512, "my-transfer");
        assert_eq!(op.label, "my-transfer");
    }

    #[test]
    fn scheduler_chain_of_three_deps() {
        let mut sched = CopyScheduler::new(SchedulePolicy::Fifo);
        let op1 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "a");
        let id1 = op1.id;
        let op2 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "b").depends_on(id1);
        let id2 = op2.id;
        let op3 = AsyncCopyOp::new(CopyDirection::HostToDevice, 100, "c").depends_on(id2);

        sched.submit(op1);
        sched.submit(op2);
        sched.submit(op3);

        // Round 1: only op1.
        let b1 = sched.next_batch();
        assert_eq!(b1.ops.len(), 1);
        assert_eq!(b1.ops[0].label, "a");
        sched.mark_completed(b1.ops[0].clone());

        // Round 2: only op2.
        let b2 = sched.next_batch();
        assert_eq!(b2.ops.len(), 1);
        assert_eq!(b2.ops[0].label, "b");
        sched.mark_completed(b2.ops[0].clone());

        // Round 3: op3.
        let b3 = sched.next_batch();
        assert_eq!(b3.ops.len(), 1);
        assert_eq!(b3.ops[0].label, "c");
    }

    #[test]
    fn pinned_mgr_multiple_alloc_free_cycles() {
        let config = PinnedMemoryConfig {
            max_pinned_bytes: 32768,
            preallocate_sizes: Vec::new(),
            alignment: 4096,
        };
        let mut mgr = PinnedMemoryManager::new(config).unwrap();
        for _ in 0..10 {
            let id = mgr.allocate(4096, "cycle").unwrap();
            mgr.free(id).unwrap();
        }
        // All reuse the same block.
        assert_eq!(mgr.num_allocations(), 1);
    }

    #[test]
    fn staged_plan_large_transfer_many_chunks() {
        let plan = StagedTransferPlan::plan(
            1_000_000,
            1024,
            CopyDirection::HostToDevice,
            StagingStrategy::DoubleBuffered,
        )
        .unwrap();
        assert_eq!(plan.num_chunks(), 1_000_000usize.div_ceil(1024));
        // Verify total bytes covered.
        let total: usize = plan.chunks.iter().map(|c| c.size).sum();
        assert_eq!(total, 1_000_000);
    }
}
