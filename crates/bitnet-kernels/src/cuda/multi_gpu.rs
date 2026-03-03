//! CUDA multi-GPU operations for distributed inference.
//!
//! # Overview
//!
//! Provides multi-GPU coordination primitives for data-parallel inference
//! across heterogeneous GPU topologies. Key components:
//!
//! - [`DeviceTopology`] — query and cache multi-GPU topology (P2P, NVLink).
//! - [`MultiDeviceContext`] — manage execution contexts across multiple GPUs.
//! - [`TensorSplitter`] — split tensors across devices for data parallelism.
//! - [`AllReduceOp`] — ring allreduce for gradient aggregation.
//! - [`PeerTransfer`] — GPU-to-GPU direct memory transfer.
//! - [`LoadBalancer`] — dynamic work distribution across heterogeneous GPUs.
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU fallback implementations are provided for testing on non-GPU hosts.

use bitnet_common::{KernelError, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TRANSFER_ID: AtomicU64 = AtomicU64::new(1);

fn next_context_id() -> u64 {
    NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_transfer_id() -> u64 {
    NEXT_TRANSFER_ID.fetch_add(1, Ordering::Relaxed)
}

// ── DeviceId ─────────────────────────────────────────────────────────

/// Opaque identifier for a GPU device in the topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(pub u32);

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GPU:{}", self.0)
    }
}

// ── InterconnectType ─────────────────────────────────────────────────

/// Type of interconnect between two GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterconnectType {
    /// No direct connection — transfers go through host memory.
    None,
    /// PCIe peer-to-peer connection.
    Pcie,
    /// NVLink high-bandwidth interconnect.
    NvLink,
    /// NVSwitch fabric (fully connected NVLink).
    NvSwitch,
}

impl InterconnectType {
    /// Estimated unidirectional bandwidth in GB/s for this interconnect.
    pub fn bandwidth_gbps(self) -> f64 {
        match self {
            Self::None => 12.0,    // PCIe host staging
            Self::Pcie => 25.0,    // PCIe Gen4 x16
            Self::NvLink => 300.0, // NVLink 3.0 per link
            Self::NvSwitch => 600.0,
        }
    }
}

// ── PeerLink ─────────────────────────────────────────────────────────

/// Describes the link between two devices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeerLink {
    /// Source device.
    pub src: DeviceId,
    /// Destination device.
    pub dst: DeviceId,
    /// Interconnect type.
    pub interconnect: InterconnectType,
    /// Measured or estimated bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Measured latency in microseconds (0 if unknown).
    pub latency_us: f64,
}

// ── DeviceInfo ───────────────────────────────────────────────────────

/// Static properties of a single GPU.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device ordinal.
    pub id: DeviceId,
    /// Human-readable name (e.g. "NVIDIA A100-SXM4-80GB").
    pub name: String,
    /// Total global memory in bytes.
    pub total_memory: usize,
    /// CUDA compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Maximum number of threads per SM.
    pub max_threads_per_sm: u32,
}

impl DeviceInfo {
    /// Compute a rough FLOPS estimate for load-balancing heuristics.
    pub fn estimated_flops(&self) -> f64 {
        // Simplified: SM count × max threads × 2 (FMA) × clock (assume 1.5 GHz).
        self.sm_count as f64 * self.max_threads_per_sm as f64 * 2.0 * 1.5e9
    }
}

// ── DeviceTopology ───────────────────────────────────────────────────

/// Cached multi-GPU topology graph.
///
/// On real CUDA hardware this would call `cudaDeviceCanAccessPeer` and
/// `nvmlDeviceGetTopologyCommonAncestor`. The CPU fallback populates a
/// synthetic topology for testing.
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    devices: Vec<DeviceInfo>,
    /// `links[(src, dst)]` → PeerLink.
    links: HashMap<(DeviceId, DeviceId), PeerLink>,
    /// Timestamp when topology was last refreshed.
    refreshed_at: Instant,
}

impl DeviceTopology {
    /// Discover topology for the given device count (CPU fallback).
    pub fn discover(num_devices: u32) -> Result<Self> {
        if num_devices == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_devices must be >= 1".into(),
            }
            .into());
        }
        let mut devices = Vec::with_capacity(num_devices as usize);
        for i in 0..num_devices {
            devices.push(DeviceInfo {
                id: DeviceId(i),
                name: format!("CPU-Fallback-Device-{i}"),
                total_memory: 8 * 1024 * 1024 * 1024, // 8 GiB
                compute_capability: (8, 0),
                sm_count: 108,
                max_threads_per_sm: 2048,
            });
        }

        let mut links = HashMap::new();
        for i in 0..num_devices {
            for j in 0..num_devices {
                if i == j {
                    continue;
                }
                let interconnect = if (i as i32 - j as i32).unsigned_abs() == 1 {
                    InterconnectType::NvLink
                } else {
                    InterconnectType::Pcie
                };
                links.insert(
                    (DeviceId(i), DeviceId(j)),
                    PeerLink {
                        src: DeviceId(i),
                        dst: DeviceId(j),
                        interconnect,
                        bandwidth_gbps: interconnect.bandwidth_gbps(),
                        latency_us: match interconnect {
                            InterconnectType::NvLink => 1.0,
                            _ => 5.0,
                        },
                    },
                );
            }
        }

        Ok(Self { devices, links, refreshed_at: Instant::now() })
    }

    /// Build a topology from explicit device info and links.
    pub fn from_parts(devices: Vec<DeviceInfo>, links: Vec<PeerLink>) -> Result<Self> {
        if devices.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "devices must not be empty".into(),
            }
            .into());
        }
        let link_map: HashMap<_, _> = links.into_iter().map(|l| ((l.src, l.dst), l)).collect();
        Ok(Self { devices, links: link_map, refreshed_at: Instant::now() })
    }

    /// Number of devices in the topology.
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get device info by ordinal.
    pub fn device(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// Iterate over all devices.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Get the link between two devices (if any).
    pub fn link(&self, src: DeviceId, dst: DeviceId) -> Option<&PeerLink> {
        self.links.get(&(src, dst))
    }

    /// Check whether two devices have direct P2P access.
    pub fn has_p2p(&self, a: DeviceId, b: DeviceId) -> bool {
        self.links.get(&(a, b)).is_some_and(|l| l.interconnect != InterconnectType::None)
    }

    /// Check whether two devices are connected via NVLink.
    pub fn has_nvlink(&self, a: DeviceId, b: DeviceId) -> bool {
        self.links.get(&(a, b)).is_some_and(|l| {
            matches!(l.interconnect, InterconnectType::NvLink | InterconnectType::NvSwitch)
        })
    }

    /// Total aggregated bandwidth across all links from a given device (GB/s).
    pub fn aggregate_bandwidth(&self, src: DeviceId) -> f64 {
        self.links.values().filter(|l| l.src == src).map(|l| l.bandwidth_gbps).sum()
    }

    /// Duration since last topology refresh.
    pub fn age(&self) -> Duration {
        self.refreshed_at.elapsed()
    }

    /// Compute the optimal ring order for allreduce by maximising link
    /// bandwidth along consecutive pairs.
    pub fn optimal_ring_order(&self) -> Vec<DeviceId> {
        let n = self.devices.len();
        if n <= 1 {
            return self.devices.iter().map(|d| d.id).collect();
        }
        // Greedy nearest-neighbour heuristic.
        let mut visited = vec![false; n];
        let mut ring = Vec::with_capacity(n);
        ring.push(DeviceId(0));
        visited[0] = true;
        for _ in 1..n {
            let cur = *ring.last().unwrap();
            let next = (0..n)
                .filter(|&i| !visited[i])
                .max_by(|&a, &b| {
                    let bw_a = self
                        .links
                        .get(&(cur, DeviceId(a as u32)))
                        .map_or(0.0, |l| l.bandwidth_gbps);
                    let bw_b = self
                        .links
                        .get(&(cur, DeviceId(b as u32)))
                        .map_or(0.0, |l| l.bandwidth_gbps);
                    bw_a.partial_cmp(&bw_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            visited[next] = true;
            ring.push(DeviceId(next as u32));
        }
        ring
    }
}

// ── DeviceContextState ───────────────────────────────────────────────

/// State of a per-device execution context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceContextState {
    /// Not yet initialised.
    Uninitialised,
    /// Ready for kernel dispatch.
    Ready,
    /// Currently executing kernels.
    Busy,
    /// Encountered an error.
    Error,
}

// ── PerDeviceContext ─────────────────────────────────────────────────

/// Per-device execution context (CPU fallback).
#[derive(Debug)]
pub struct PerDeviceContext {
    pub device_id: DeviceId,
    pub state: DeviceContextState,
    /// Bytes currently allocated on this device.
    pub allocated_bytes: usize,
    /// Number of kernels dispatched.
    pub kernels_dispatched: u64,
    /// Creation time.
    created_at: Instant,
}

impl PerDeviceContext {
    fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            state: DeviceContextState::Ready,
            allocated_bytes: 0,
            kernels_dispatched: 0,
            created_at: Instant::now(),
        }
    }

    /// Uptime of this context.
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ── MultiDeviceContext ───────────────────────────────────────────────

/// Manages execution contexts across multiple GPUs.
///
/// Provides a unified interface for allocating memory, dispatching kernels,
/// and synchronising across devices.
#[derive(Debug)]
pub struct MultiDeviceContext {
    /// Unique context id.
    pub id: u64,
    /// Underlying topology.
    topology: DeviceTopology,
    /// Per-device contexts, keyed by ordinal.
    contexts: HashMap<DeviceId, PerDeviceContext>,
    /// Whether the context has been synchronised.
    synced: bool,
}

impl MultiDeviceContext {
    /// Create contexts for all devices in the topology.
    pub fn new(topology: DeviceTopology) -> Self {
        let contexts: HashMap<_, _> =
            topology.devices().iter().map(|d| (d.id, PerDeviceContext::new(d.id))).collect();
        Self { id: next_context_id(), topology, contexts, synced: true }
    }

    /// Number of managed devices.
    pub fn num_devices(&self) -> usize {
        self.contexts.len()
    }

    /// Get the underlying topology.
    pub fn topology(&self) -> &DeviceTopology {
        &self.topology
    }

    /// Get a per-device context.
    pub fn device_ctx(&self, id: DeviceId) -> Option<&PerDeviceContext> {
        self.contexts.get(&id)
    }

    /// Get a mutable per-device context.
    pub fn device_ctx_mut(&mut self, id: DeviceId) -> Option<&mut PerDeviceContext> {
        self.contexts.get_mut(&id)
    }

    /// Record an allocation on a device.
    pub fn allocate(&mut self, device: DeviceId, bytes: usize) -> Result<()> {
        let ctx = self.contexts.get_mut(&device).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("unknown device {device}"),
        })?;
        let info = self.topology.device(device).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("device {device} not in topology"),
        })?;
        if ctx.allocated_bytes + bytes > info.total_memory {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "OOM on {device}: need {bytes} B, have {} B free",
                    info.total_memory - ctx.allocated_bytes
                ),
            }
            .into());
        }
        ctx.allocated_bytes += bytes;
        self.synced = false;
        Ok(())
    }

    /// Free memory on a device.
    pub fn free(&mut self, device: DeviceId, bytes: usize) -> Result<()> {
        let ctx = self.contexts.get_mut(&device).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("unknown device {device}"),
        })?;
        ctx.allocated_bytes = ctx.allocated_bytes.saturating_sub(bytes);
        Ok(())
    }

    /// Record kernel dispatch.
    pub fn dispatch_kernel(&mut self, device: DeviceId) -> Result<()> {
        let ctx = self.contexts.get_mut(&device).ok_or_else(|| KernelError::InvalidArguments {
            reason: format!("unknown device {device}"),
        })?;
        if ctx.state == DeviceContextState::Error {
            return Err(KernelError::InvalidArguments {
                reason: format!("device {device} is in error state"),
            }
            .into());
        }
        ctx.state = DeviceContextState::Busy;
        ctx.kernels_dispatched += 1;
        self.synced = false;
        Ok(())
    }

    /// Synchronise all devices (CPU fallback: mark as synced).
    pub fn sync_all(&mut self) -> Result<()> {
        for ctx in self.contexts.values_mut() {
            if ctx.state == DeviceContextState::Busy {
                ctx.state = DeviceContextState::Ready;
            }
        }
        self.synced = true;
        Ok(())
    }

    /// Whether all devices are currently synchronised.
    pub fn is_synced(&self) -> bool {
        self.synced
    }

    /// Total memory allocated across all devices.
    pub fn total_allocated(&self) -> usize {
        self.contexts.values().map(|c| c.allocated_bytes).sum()
    }

    /// Reset all contexts to initial state, freeing tracked memory.
    pub fn reset(&mut self) {
        for ctx in self.contexts.values_mut() {
            ctx.allocated_bytes = 0;
            ctx.kernels_dispatched = 0;
            ctx.state = DeviceContextState::Ready;
        }
        self.synced = true;
    }
}

// ── SplitStrategy ────────────────────────────────────────────────────

/// Strategy for distributing tensor chunks across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitStrategy {
    /// Equal-sized chunks (pad the last chunk if necessary).
    Equal,
    /// Proportional to device FLOPS estimate.
    Proportional,
    /// Proportional to available (free) memory.
    MemoryProportional,
}

// ── TensorChunk ──────────────────────────────────────────────────────

/// Metadata for a chunk of a split tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorChunk {
    /// Target device.
    pub device: DeviceId,
    /// Start offset in the original tensor (in elements).
    pub offset: usize,
    /// Number of elements in this chunk.
    pub length: usize,
    /// Size in bytes.
    pub size_bytes: usize,
}

// ── TensorSplitter ───────────────────────────────────────────────────

/// Splits tensors across devices for data parallelism.
#[derive(Debug)]
pub struct TensorSplitter {
    topology: DeviceTopology,
    strategy: SplitStrategy,
}

impl TensorSplitter {
    /// Create a new splitter with the given strategy.
    pub fn new(topology: DeviceTopology, strategy: SplitStrategy) -> Self {
        Self { topology, strategy }
    }

    /// Split `total_elements` across all devices. `element_size` is in bytes.
    pub fn split(&self, total_elements: usize, element_size: usize) -> Result<Vec<TensorChunk>> {
        let n = self.topology.num_devices();
        if n == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "no devices in topology".into() }.into()
            );
        }
        if total_elements == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "total_elements must be > 0".into(),
            }
            .into());
        }

        let weights = self.compute_weights();
        let mut chunks = Vec::with_capacity(n);
        let mut offset = 0usize;
        let total_weight: f64 = weights.iter().sum();

        for (i, &w) in weights.iter().enumerate() {
            let len = if i == n - 1 {
                // Last device gets the remainder.
                total_elements - offset
            } else {
                ((total_elements as f64) * (w / total_weight)).round() as usize
            };
            let len = len.max(1).min(total_elements - offset);
            chunks.push(TensorChunk {
                device: DeviceId(i as u32),
                offset,
                length: len,
                size_bytes: len * element_size,
            });
            offset += len;
            if offset >= total_elements {
                break;
            }
        }
        Ok(chunks)
    }

    /// Split along a specific axis. Returns chunk metadata per device.
    pub fn split_axis(
        &self,
        shape: &[usize],
        axis: usize,
        element_size: usize,
    ) -> Result<Vec<TensorChunk>> {
        if axis >= shape.len() {
            return Err(KernelError::InvalidArguments {
                reason: format!("axis {axis} out of bounds for tensor with {} dims", shape.len()),
            }
            .into());
        }
        let axis_size = shape[axis];
        let elements_per_slice: usize = shape.iter().skip(axis + 1).product::<usize>().max(1);
        let inner_chunks = self.split(axis_size, element_size * elements_per_slice)?;
        Ok(inner_chunks)
    }

    /// Change the split strategy.
    pub fn set_strategy(&mut self, strategy: SplitStrategy) {
        self.strategy = strategy;
    }

    /// Current strategy.
    pub fn strategy(&self) -> SplitStrategy {
        self.strategy
    }

    fn compute_weights(&self) -> Vec<f64> {
        let devices = self.topology.devices();
        match self.strategy {
            SplitStrategy::Equal => vec![1.0; devices.len()],
            SplitStrategy::Proportional => devices.iter().map(|d| d.estimated_flops()).collect(),
            SplitStrategy::MemoryProportional => {
                devices.iter().map(|d| d.total_memory as f64).collect()
            }
        }
    }
}

// ── ReduceOp ─────────────────────────────────────────────────────────

/// Operation to perform during allreduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise average.
    Average,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

// ── AllReduceConfig ──────────────────────────────────────────────────

/// Configuration for an allreduce operation.
#[derive(Debug, Clone)]
pub struct AllReduceConfig {
    /// Reduction operation.
    pub op: ReduceOp,
    /// Ring order (device ordinals); if empty, uses natural order.
    pub ring_order: Vec<DeviceId>,
    /// Chunk size for pipelining (elements). 0 = single chunk.
    pub chunk_size: usize,
}

impl Default for AllReduceConfig {
    fn default() -> Self {
        Self { op: ReduceOp::Sum, ring_order: Vec::new(), chunk_size: 0 }
    }
}

// ── AllReduceOp ──────────────────────────────────────────────────────

/// Ring allreduce for gradient aggregation (CPU fallback).
///
/// On real hardware this would use NCCL or a custom ring kernel.
/// The CPU fallback simulates the reduce-scatter + allgather phases.
#[derive(Debug)]
pub struct AllReduceOp {
    config: AllReduceConfig,
    num_devices: usize,
    /// Allreduce invocation counter.
    invocations: u64,
}

impl AllReduceOp {
    /// Create a new allreduce operator.
    pub fn new(config: AllReduceConfig, num_devices: usize) -> Result<Self> {
        if num_devices == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "num_devices must be >= 1".into(),
            }
            .into());
        }
        Ok(Self { config, num_devices, invocations: 0 })
    }

    /// Perform an in-place allreduce across per-device buffers.
    ///
    /// `buffers[i]` is the local buffer on device `i`. All buffers must
    /// have the same length. After return every buffer holds the reduced
    /// result.
    pub fn execute(&mut self, buffers: &mut [Vec<f32>]) -> Result<()> {
        if buffers.len() != self.num_devices {
            return Err(KernelError::InvalidArguments {
                reason: format!("expected {} buffers, got {}", self.num_devices, buffers.len()),
            }
            .into());
        }
        let n = buffers[0].len();
        if buffers.iter().any(|b| b.len() != n) {
            return Err(KernelError::InvalidArguments {
                reason: "all buffers must have the same length".into(),
            }
            .into());
        }
        if n == 0 {
            self.invocations += 1;
            return Ok(());
        }

        let ring = self.effective_ring();
        let num_devs = ring.len();

        // Phase 1: Reduce-scatter — each device accumulates a segment.
        let chunk_len = (n + num_devs - 1) / num_devs;
        let mut reduced = vec![0.0f32; n];

        for elem in 0..n {
            let val: f32 = (0..num_devs)
                .map(|d| buffers[d][elem])
                .collect::<Vec<_>>()
                .iter()
                .copied()
                .reduce(|a, b| self.apply_op(a, b))
                .unwrap_or(0.0);
            reduced[elem] =
                if self.config.op == ReduceOp::Average { val / num_devs as f32 } else { val };
        }

        // Phase 2: Allgather — broadcast the result to every device.
        for buf in buffers.iter_mut() {
            buf.copy_from_slice(&reduced);
        }

        self.invocations += 1;
        Ok(())
    }

    /// Number of allreduce invocations.
    pub fn invocations(&self) -> u64 {
        self.invocations
    }

    /// Current configuration.
    pub fn config(&self) -> &AllReduceConfig {
        &self.config
    }

    fn effective_ring(&self) -> Vec<usize> {
        if self.config.ring_order.is_empty() {
            (0..self.num_devices).collect()
        } else {
            self.config.ring_order.iter().map(|d| d.0 as usize).collect()
        }
    }

    fn apply_op(&self, a: f32, b: f32) -> f32 {
        match self.config.op {
            ReduceOp::Sum | ReduceOp::Average => a + b,
            ReduceOp::Max => a.max(b),
            ReduceOp::Min => a.min(b),
        }
    }
}

// ── TransferDirection ────────────────────────────────────────────────

/// Direction of a peer memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Device-to-device.
    DeviceToDevice,
    /// Host-to-device.
    HostToDevice,
    /// Device-to-host.
    DeviceToHost,
}

// ── TransferStatus ───────────────────────────────────────────────────

/// Status of a peer transfer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    /// Queued but not started.
    Pending,
    /// Currently in flight.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
}

// ── TransferRecord ───────────────────────────────────────────────────

/// Record of a completed (or in-flight) transfer.
#[derive(Debug, Clone)]
pub struct TransferRecord {
    /// Unique transfer id.
    pub id: u64,
    /// Source device.
    pub src: DeviceId,
    /// Destination device.
    pub dst: DeviceId,
    /// Direction.
    pub direction: TransferDirection,
    /// Number of bytes transferred.
    pub bytes: usize,
    /// Status.
    pub status: TransferStatus,
    /// Time when the transfer was initiated.
    pub initiated_at: Instant,
    /// Duration of the transfer (if completed).
    pub duration: Option<Duration>,
}

// ── PeerTransfer ─────────────────────────────────────────────────────

/// GPU-to-GPU direct memory transfer manager (CPU fallback).
///
/// On real hardware this would use `cudaMemcpyPeerAsync` with P2P
/// enabled device pairs. The CPU fallback records transfer metadata
/// and simulates data movement.
#[derive(Debug)]
pub struct PeerTransfer {
    topology: DeviceTopology,
    history: Vec<TransferRecord>,
    total_bytes: u64,
}

impl PeerTransfer {
    /// Create a new peer transfer manager.
    pub fn new(topology: DeviceTopology) -> Self {
        Self { topology, history: Vec::new(), total_bytes: 0 }
    }

    /// Simulate a device-to-device transfer. `src_data` is copied into
    /// `dst_data`.
    pub fn transfer(
        &mut self,
        src: DeviceId,
        dst: DeviceId,
        src_data: &[f32],
        dst_data: &mut [f32],
    ) -> Result<TransferRecord> {
        if src_data.len() != dst_data.len() {
            return Err(KernelError::InvalidArguments {
                reason: "src and dst buffers must have the same length".into(),
            }
            .into());
        }
        let bytes = src_data.len() * std::mem::size_of::<f32>();
        let start = Instant::now();
        dst_data.copy_from_slice(src_data);
        let dur = start.elapsed();

        let rec = TransferRecord {
            id: next_transfer_id(),
            src,
            dst,
            direction: TransferDirection::DeviceToDevice,
            bytes,
            status: TransferStatus::Completed,
            initiated_at: start,
            duration: Some(dur),
        };
        self.total_bytes += bytes as u64;
        self.history.push(rec.clone());
        Ok(rec)
    }

    /// Initiate a zero-copy transfer record (metadata only, no data movement).
    pub fn record_transfer(
        &mut self,
        src: DeviceId,
        dst: DeviceId,
        bytes: usize,
        direction: TransferDirection,
    ) -> TransferRecord {
        let rec = TransferRecord {
            id: next_transfer_id(),
            src,
            dst,
            direction,
            bytes,
            status: TransferStatus::Completed,
            initiated_at: Instant::now(),
            duration: Some(Duration::ZERO),
        };
        self.total_bytes += bytes as u64;
        self.history.push(rec.clone());
        rec
    }

    /// Total bytes transferred.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Number of transfers recorded.
    pub fn num_transfers(&self) -> usize {
        self.history.len()
    }

    /// Transfer history.
    pub fn history(&self) -> &[TransferRecord] {
        &self.history
    }

    /// Estimated bandwidth between two devices.
    pub fn estimated_bandwidth(&self, src: DeviceId, dst: DeviceId) -> f64 {
        self.topology.link(src, dst).map_or(12.0, |l| l.bandwidth_gbps)
    }

    /// Clear transfer history.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.total_bytes = 0;
    }
}

// ── LoadBalancerStrategy ─────────────────────────────────────────────

/// Strategy for distributing work across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancerStrategy {
    /// Round-robin across devices.
    RoundRobin,
    /// Route to least-loaded device (fewest dispatched kernels).
    LeastLoaded,
    /// Route proportionally to device compute capacity.
    WeightedCapacity,
    /// Route to device with most free memory.
    MemoryAware,
}

// ── WorkItem ─────────────────────────────────────────────────────────

/// A unit of work to be dispatched to a device.
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Human-readable label.
    pub label: String,
    /// Estimated compute cost (arbitrary units).
    pub cost: u64,
    /// Memory requirement in bytes.
    pub memory_bytes: usize,
}

impl WorkItem {
    /// Create a new work item.
    pub fn new(label: impl Into<String>, cost: u64, memory_bytes: usize) -> Self {
        Self { label: label.into(), cost, memory_bytes }
    }
}

// ── Assignment ───────────────────────────────────────────────────────

/// Binding of a work item to a specific device.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// The work item.
    pub item: WorkItem,
    /// Target device.
    pub device: DeviceId,
}

// ── LoadBalancer ─────────────────────────────────────────────────────

/// Dynamic work distribution across heterogeneous GPUs.
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancerStrategy,
    topology: DeviceTopology,
    /// Accumulated cost per device.
    device_load: HashMap<DeviceId, u64>,
    /// Memory used per device.
    device_memory: HashMap<DeviceId, usize>,
    /// Round-robin counter.
    rr_index: usize,
    /// Total items dispatched.
    dispatched: u64,
}

impl LoadBalancer {
    /// Create a new load balancer.
    pub fn new(topology: DeviceTopology, strategy: LoadBalancerStrategy) -> Self {
        let device_load: HashMap<_, _> = topology.devices().iter().map(|d| (d.id, 0u64)).collect();
        let device_memory: HashMap<_, _> =
            topology.devices().iter().map(|d| (d.id, 0usize)).collect();
        Self { strategy, topology, device_load, device_memory, rr_index: 0, dispatched: 0 }
    }

    /// Assign a single work item to the best device.
    pub fn assign(&mut self, item: WorkItem) -> Result<Assignment> {
        let device = self.select_device(&item)?;
        *self.device_load.get_mut(&device).unwrap() += item.cost;
        *self.device_memory.get_mut(&device).unwrap() += item.memory_bytes;
        self.dispatched += 1;
        Ok(Assignment { item, device })
    }

    /// Assign a batch of work items.
    pub fn assign_batch(&mut self, items: Vec<WorkItem>) -> Result<Vec<Assignment>> {
        items.into_iter().map(|item| self.assign(item)).collect()
    }

    /// Get load on a specific device.
    pub fn device_load(&self, id: DeviceId) -> u64 {
        self.device_load.get(&id).copied().unwrap_or(0)
    }

    /// Get memory usage on a specific device.
    pub fn device_memory_used(&self, id: DeviceId) -> usize {
        self.device_memory.get(&id).copied().unwrap_or(0)
    }

    /// Total items dispatched.
    pub fn total_dispatched(&self) -> u64 {
        self.dispatched
    }

    /// Current strategy.
    pub fn strategy(&self) -> LoadBalancerStrategy {
        self.strategy
    }

    /// Change the strategy.
    pub fn set_strategy(&mut self, strategy: LoadBalancerStrategy) {
        self.strategy = strategy;
    }

    /// Reset all tracked load.
    pub fn reset(&mut self) {
        for v in self.device_load.values_mut() {
            *v = 0;
        }
        for v in self.device_memory.values_mut() {
            *v = 0;
        }
        self.rr_index = 0;
        self.dispatched = 0;
    }

    /// Load imbalance ratio: max_load / mean_load. Returns 1.0 for
    /// perfectly balanced, higher is worse.
    pub fn imbalance_ratio(&self) -> f64 {
        let loads: Vec<f64> = self.device_load.values().map(|&v| v as f64).collect();
        let mean = loads.iter().sum::<f64>() / loads.len() as f64;
        if mean == 0.0 {
            return 1.0;
        }
        let max = loads.iter().copied().fold(0.0f64, f64::max);
        max / mean
    }

    fn select_device(&mut self, item: &WorkItem) -> Result<DeviceId> {
        let devices = self.topology.devices();
        if devices.is_empty() {
            return Err(
                KernelError::InvalidArguments { reason: "no devices available".into() }.into()
            );
        }
        match self.strategy {
            LoadBalancerStrategy::RoundRobin => {
                let dev = devices[self.rr_index % devices.len()].id;
                self.rr_index += 1;
                Ok(dev)
            }
            LoadBalancerStrategy::LeastLoaded => {
                let best = devices
                    .iter()
                    .min_by_key(|d| self.device_load.get(&d.id).copied().unwrap_or(0))
                    .unwrap();
                Ok(best.id)
            }
            LoadBalancerStrategy::WeightedCapacity => {
                // Route to device with the lowest load-to-capacity ratio.
                let best = devices
                    .iter()
                    .min_by(|a, b| {
                        let ratio_a = self.device_load.get(&a.id).copied().unwrap_or(0) as f64
                            / a.estimated_flops().max(1.0);
                        let ratio_b = self.device_load.get(&b.id).copied().unwrap_or(0) as f64
                            / b.estimated_flops().max(1.0);
                        ratio_a.partial_cmp(&ratio_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                Ok(best.id)
            }
            LoadBalancerStrategy::MemoryAware => {
                let best = devices
                    .iter()
                    .filter(|d| {
                        let used = self.device_memory.get(&d.id).copied().unwrap_or(0);
                        used + item.memory_bytes <= d.total_memory
                    })
                    .min_by_key(|d| self.device_memory.get(&d.id).copied().unwrap_or(0))
                    .ok_or_else(|| KernelError::InvalidArguments {
                        reason: "no device has sufficient free memory".into(),
                    })?;
                Ok(best.id)
            }
        }
    }
}

// ── Convenience constructors ─────────────────────────────────────────

/// Create a multi-device context for `n` synthetic GPUs.
pub fn create_multi_device_context(num_devices: u32) -> Result<MultiDeviceContext> {
    let topo = DeviceTopology::discover(num_devices)?;
    Ok(MultiDeviceContext::new(topo))
}

/// Perform an allreduce sum across `buffers` using a default ring.
pub fn allreduce_sum(buffers: &mut [Vec<f32>]) -> Result<()> {
    let mut op = AllReduceOp::new(AllReduceConfig::default(), buffers.len())?;
    op.execute(buffers)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DeviceId ─────────────────────────────────────────────────────

    #[test]
    fn device_id_display() {
        assert_eq!(format!("{}", DeviceId(0)), "GPU:0");
        assert_eq!(format!("{}", DeviceId(7)), "GPU:7");
    }

    #[test]
    fn device_id_ordering() {
        assert!(DeviceId(0) < DeviceId(1));
        assert_eq!(DeviceId(3), DeviceId(3));
    }

    // ── InterconnectType ─────────────────────────────────────────────

    #[test]
    fn interconnect_bandwidth_ordering() {
        assert!(InterconnectType::None.bandwidth_gbps() < InterconnectType::Pcie.bandwidth_gbps());
        assert!(
            InterconnectType::Pcie.bandwidth_gbps() < InterconnectType::NvLink.bandwidth_gbps()
        );
        assert!(
            InterconnectType::NvLink.bandwidth_gbps() < InterconnectType::NvSwitch.bandwidth_gbps()
        );
    }

    #[test]
    fn interconnect_bandwidth_positive() {
        for ic in [
            InterconnectType::None,
            InterconnectType::Pcie,
            InterconnectType::NvLink,
            InterconnectType::NvSwitch,
        ] {
            assert!(ic.bandwidth_gbps() > 0.0);
        }
    }

    // ── DeviceInfo ───────────────────────────────────────────────────

    #[test]
    fn device_info_estimated_flops_positive() {
        let info = DeviceInfo {
            id: DeviceId(0),
            name: "TestGPU".into(),
            total_memory: 1024,
            compute_capability: (8, 0),
            sm_count: 108,
            max_threads_per_sm: 2048,
        };
        assert!(info.estimated_flops() > 0.0);
    }

    // ── DeviceTopology ───────────────────────────────────────────────

    #[test]
    fn topology_discover_single() {
        let t = DeviceTopology::discover(1).unwrap();
        assert_eq!(t.num_devices(), 1);
        assert!(t.devices()[0].name.contains("Fallback"));
    }

    #[test]
    fn topology_discover_zero_fails() {
        assert!(DeviceTopology::discover(0).is_err());
    }

    #[test]
    fn topology_discover_four() {
        let t = DeviceTopology::discover(4).unwrap();
        assert_eq!(t.num_devices(), 4);
        for i in 0..4 {
            assert!(t.device(DeviceId(i)).is_some());
        }
    }

    #[test]
    fn topology_links_adjacent_nvlink() {
        let t = DeviceTopology::discover(4).unwrap();
        assert!(t.has_nvlink(DeviceId(0), DeviceId(1)));
        assert!(t.has_nvlink(DeviceId(1), DeviceId(2)));
    }

    #[test]
    fn topology_links_non_adjacent_pcie() {
        let t = DeviceTopology::discover(4).unwrap();
        // GPU 0 → GPU 2 are not adjacent.
        assert!(t.has_p2p(DeviceId(0), DeviceId(2)));
        assert!(!t.has_nvlink(DeviceId(0), DeviceId(2)));
    }

    #[test]
    fn topology_no_self_link() {
        let t = DeviceTopology::discover(2).unwrap();
        assert!(t.link(DeviceId(0), DeviceId(0)).is_none());
    }

    #[test]
    fn topology_aggregate_bandwidth() {
        let t = DeviceTopology::discover(3).unwrap();
        let bw = t.aggregate_bandwidth(DeviceId(0));
        assert!(bw > 0.0);
    }

    #[test]
    fn topology_age_is_recent() {
        let t = DeviceTopology::discover(1).unwrap();
        assert!(t.age() < Duration::from_secs(1));
    }

    #[test]
    fn topology_from_parts_empty_fails() {
        assert!(DeviceTopology::from_parts(vec![], vec![]).is_err());
    }

    #[test]
    fn topology_from_parts_custom() {
        let d = DeviceInfo {
            id: DeviceId(0),
            name: "Custom".into(),
            total_memory: 1024,
            compute_capability: (9, 0),
            sm_count: 1,
            max_threads_per_sm: 1,
        };
        let t = DeviceTopology::from_parts(vec![d], vec![]).unwrap();
        assert_eq!(t.num_devices(), 1);
        assert_eq!(t.device(DeviceId(0)).unwrap().name, "Custom");
    }

    #[test]
    fn topology_optimal_ring_single() {
        let t = DeviceTopology::discover(1).unwrap();
        let ring = t.optimal_ring_order();
        assert_eq!(ring, vec![DeviceId(0)]);
    }

    #[test]
    fn topology_optimal_ring_multi() {
        let t = DeviceTopology::discover(4).unwrap();
        let ring = t.optimal_ring_order();
        assert_eq!(ring.len(), 4);
        // All devices should appear exactly once.
        let mut sorted = ring.clone();
        sorted.sort();
        assert_eq!(sorted, vec![DeviceId(0), DeviceId(1), DeviceId(2), DeviceId(3)]);
    }

    // ── MultiDeviceContext ───────────────────────────────────────────

    #[test]
    fn context_create_and_num_devices() {
        let ctx = create_multi_device_context(4).unwrap();
        assert_eq!(ctx.num_devices(), 4);
    }

    #[test]
    fn context_initially_synced() {
        let ctx = create_multi_device_context(2).unwrap();
        assert!(ctx.is_synced());
    }

    #[test]
    fn context_allocate_and_free() {
        let mut ctx = create_multi_device_context(2).unwrap();
        ctx.allocate(DeviceId(0), 1024).unwrap();
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().allocated_bytes, 1024);
        ctx.free(DeviceId(0), 512).unwrap();
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().allocated_bytes, 512);
    }

    #[test]
    fn context_allocate_oom() {
        let mut ctx = create_multi_device_context(1).unwrap();
        let total = ctx.topology().device(DeviceId(0)).unwrap().total_memory;
        assert!(ctx.allocate(DeviceId(0), total + 1).is_err());
    }

    #[test]
    fn context_allocate_unknown_device() {
        let mut ctx = create_multi_device_context(1).unwrap();
        assert!(ctx.allocate(DeviceId(99), 1).is_err());
    }

    #[test]
    fn context_dispatch_kernel() {
        let mut ctx = create_multi_device_context(2).unwrap();
        ctx.dispatch_kernel(DeviceId(0)).unwrap();
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().kernels_dispatched, 1);
        assert!(!ctx.is_synced());
    }

    #[test]
    fn context_dispatch_error_state() {
        let mut ctx = create_multi_device_context(1).unwrap();
        ctx.device_ctx_mut(DeviceId(0)).unwrap().state = DeviceContextState::Error;
        assert!(ctx.dispatch_kernel(DeviceId(0)).is_err());
    }

    #[test]
    fn context_sync_all() {
        let mut ctx = create_multi_device_context(2).unwrap();
        ctx.dispatch_kernel(DeviceId(0)).unwrap();
        ctx.dispatch_kernel(DeviceId(1)).unwrap();
        ctx.sync_all().unwrap();
        assert!(ctx.is_synced());
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().state, DeviceContextState::Ready);
    }

    #[test]
    fn context_total_allocated() {
        let mut ctx = create_multi_device_context(2).unwrap();
        ctx.allocate(DeviceId(0), 100).unwrap();
        ctx.allocate(DeviceId(1), 200).unwrap();
        assert_eq!(ctx.total_allocated(), 300);
    }

    #[test]
    fn context_reset() {
        let mut ctx = create_multi_device_context(2).unwrap();
        ctx.allocate(DeviceId(0), 100).unwrap();
        ctx.dispatch_kernel(DeviceId(0)).unwrap();
        ctx.reset();
        assert_eq!(ctx.total_allocated(), 0);
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().kernels_dispatched, 0);
        assert!(ctx.is_synced());
    }

    #[test]
    fn context_uptime_positive() {
        let ctx = create_multi_device_context(1).unwrap();
        let up = ctx.device_ctx(DeviceId(0)).unwrap().uptime();
        assert!(up < Duration::from_secs(1));
    }

    #[test]
    fn context_free_saturates_at_zero() {
        let mut ctx = create_multi_device_context(1).unwrap();
        ctx.free(DeviceId(0), 9999).unwrap();
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().allocated_bytes, 0);
    }

    // ── TensorSplitter ──────────────────────────────────────────────

    #[test]
    fn splitter_equal_two_devices() {
        let topo = DeviceTopology::discover(2).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(100, 4).unwrap();
        assert_eq!(chunks.len(), 2);
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn splitter_equal_four_devices() {
        let topo = DeviceTopology::discover(4).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(1000, 4).unwrap();
        assert_eq!(chunks.len(), 4);
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn splitter_equal_odd_elements() {
        let topo = DeviceTopology::discover(3).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(10, 4).unwrap();
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn splitter_proportional() {
        let topo = DeviceTopology::discover(2).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Proportional);
        let chunks = s.split(1000, 4).unwrap();
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn splitter_memory_proportional() {
        let topo = DeviceTopology::discover(2).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::MemoryProportional);
        let chunks = s.split(500, 4).unwrap();
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn splitter_zero_elements_fails() {
        let topo = DeviceTopology::discover(1).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        assert!(s.split(0, 4).is_err());
    }

    #[test]
    fn splitter_chunk_device_ids_sequential() {
        let topo = DeviceTopology::discover(3).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(99, 4).unwrap();
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.device, DeviceId(i as u32));
        }
    }

    #[test]
    fn splitter_chunk_offsets_contiguous() {
        let topo = DeviceTopology::discover(4).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(256, 4).unwrap();
        let mut expected_offset = 0;
        for c in &chunks {
            assert_eq!(c.offset, expected_offset);
            expected_offset += c.length;
        }
    }

    #[test]
    fn splitter_size_bytes_correct() {
        let topo = DeviceTopology::discover(2).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(100, 8).unwrap();
        for c in &chunks {
            assert_eq!(c.size_bytes, c.length * 8);
        }
    }

    #[test]
    fn splitter_split_axis_0() {
        let topo = DeviceTopology::discover(2).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let shape = [16, 32, 64];
        let chunks = s.split_axis(&shape, 0, 4).unwrap();
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 16);
    }

    #[test]
    fn splitter_split_axis_out_of_bounds() {
        let topo = DeviceTopology::discover(2).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        assert!(s.split_axis(&[10, 20], 5, 4).is_err());
    }

    #[test]
    fn splitter_set_strategy() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut s = TensorSplitter::new(topo, SplitStrategy::Equal);
        assert_eq!(s.strategy(), SplitStrategy::Equal);
        s.set_strategy(SplitStrategy::Proportional);
        assert_eq!(s.strategy(), SplitStrategy::Proportional);
    }

    #[test]
    fn splitter_single_device() {
        let topo = DeviceTopology::discover(1).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(42, 4).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].length, 42);
        assert_eq!(chunks[0].offset, 0);
    }

    // ── AllReduceOp ─────────────────────────────────────────────────

    #[test]
    fn allreduce_sum_two_devices() {
        let mut bufs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        allreduce_sum(&mut bufs).unwrap();
        assert_eq!(bufs[0], vec![5.0, 7.0, 9.0]);
        assert_eq!(bufs[1], vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn allreduce_sum_four_devices() {
        let mut bufs = vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![3.0, 0.0], vec![4.0, 0.0]];
        allreduce_sum(&mut bufs).unwrap();
        for buf in &bufs {
            assert_eq!(buf[0], 10.0);
        }
    }

    #[test]
    fn allreduce_average() {
        let cfg = AllReduceConfig { op: ReduceOp::Average, ..Default::default() };
        let mut op = AllReduceOp::new(cfg, 2).unwrap();
        let mut bufs = vec![vec![10.0], vec![20.0]];
        op.execute(&mut bufs).unwrap();
        assert!((bufs[0][0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn allreduce_max() {
        let cfg = AllReduceConfig { op: ReduceOp::Max, ..Default::default() };
        let mut op = AllReduceOp::new(cfg, 3).unwrap();
        let mut bufs = vec![vec![1.0, 9.0], vec![5.0, 2.0], vec![3.0, 7.0]];
        op.execute(&mut bufs).unwrap();
        assert_eq!(bufs[0], vec![5.0, 9.0]);
        assert_eq!(bufs[1], vec![5.0, 9.0]);
    }

    #[test]
    fn allreduce_min() {
        let cfg = AllReduceConfig { op: ReduceOp::Min, ..Default::default() };
        let mut op = AllReduceOp::new(cfg, 2).unwrap();
        let mut bufs = vec![vec![3.0, 1.0], vec![1.0, 5.0]];
        op.execute(&mut bufs).unwrap();
        assert_eq!(bufs[0], vec![1.0, 1.0]);
    }

    #[test]
    fn allreduce_empty_buffers() {
        let mut op = AllReduceOp::new(AllReduceConfig::default(), 2).unwrap();
        let mut bufs = vec![vec![], vec![]];
        op.execute(&mut bufs).unwrap();
        assert_eq!(op.invocations(), 1);
    }

    #[test]
    fn allreduce_mismatched_lengths_fails() {
        let mut op = AllReduceOp::new(AllReduceConfig::default(), 2).unwrap();
        let mut bufs = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(op.execute(&mut bufs).is_err());
    }

    #[test]
    fn allreduce_wrong_device_count_fails() {
        let mut op = AllReduceOp::new(AllReduceConfig::default(), 3).unwrap();
        let mut bufs = vec![vec![1.0], vec![2.0]]; // 2 != 3
        assert!(op.execute(&mut bufs).is_err());
    }

    #[test]
    fn allreduce_zero_devices_fails() {
        assert!(AllReduceOp::new(AllReduceConfig::default(), 0).is_err());
    }

    #[test]
    fn allreduce_invocation_counter() {
        let mut op = AllReduceOp::new(AllReduceConfig::default(), 2).unwrap();
        let mut bufs = vec![vec![1.0], vec![2.0]];
        op.execute(&mut bufs).unwrap();
        op.execute(&mut bufs).unwrap();
        assert_eq!(op.invocations(), 2);
    }

    #[test]
    fn allreduce_custom_ring_order() {
        let cfg = AllReduceConfig {
            op: ReduceOp::Sum,
            ring_order: vec![DeviceId(1), DeviceId(0)],
            chunk_size: 0,
        };
        let mut op = AllReduceOp::new(cfg, 2).unwrap();
        let mut bufs = vec![vec![1.0], vec![2.0]];
        op.execute(&mut bufs).unwrap();
        assert_eq!(bufs[0], vec![3.0]);
    }

    #[test]
    fn allreduce_config_accessor() {
        let cfg = AllReduceConfig { op: ReduceOp::Max, ..Default::default() };
        let op = AllReduceOp::new(cfg, 2).unwrap();
        assert_eq!(op.config().op, ReduceOp::Max);
    }

    // ── PeerTransfer ────────────────────────────────────────────────

    #[test]
    fn transfer_device_to_device() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let src = vec![1.0, 2.0, 3.0];
        let mut dst = vec![0.0; 3];
        let rec = pt.transfer(DeviceId(0), DeviceId(1), &src, &mut dst).unwrap();
        assert_eq!(dst, vec![1.0, 2.0, 3.0]);
        assert_eq!(rec.status, TransferStatus::Completed);
        assert_eq!(rec.bytes, 12); // 3 * 4 bytes
    }

    #[test]
    fn transfer_mismatched_lengths_fails() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let src = vec![1.0, 2.0];
        let mut dst = vec![0.0; 3];
        assert!(pt.transfer(DeviceId(0), DeviceId(1), &src, &mut dst).is_err());
    }

    #[test]
    fn transfer_total_bytes_accumulates() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let src = vec![1.0; 10];
        let mut dst = vec![0.0; 10];
        pt.transfer(DeviceId(0), DeviceId(1), &src, &mut dst).unwrap();
        pt.transfer(DeviceId(1), DeviceId(0), &src, &mut dst).unwrap();
        assert_eq!(pt.total_bytes(), 80); // 2 * 10 * 4
    }

    #[test]
    fn transfer_num_transfers() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let src = vec![1.0];
        let mut dst = vec![0.0];
        pt.transfer(DeviceId(0), DeviceId(1), &src, &mut dst).unwrap();
        assert_eq!(pt.num_transfers(), 1);
    }

    #[test]
    fn transfer_history_recorded() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let src = vec![1.0; 5];
        let mut dst = vec![0.0; 5];
        pt.transfer(DeviceId(0), DeviceId(1), &src, &mut dst).unwrap();
        let h = pt.history();
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].src, DeviceId(0));
        assert_eq!(h[0].dst, DeviceId(1));
    }

    #[test]
    fn transfer_record_only() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let rec =
            pt.record_transfer(DeviceId(0), DeviceId(1), 4096, TransferDirection::HostToDevice);
        assert_eq!(rec.status, TransferStatus::Completed);
        assert_eq!(rec.bytes, 4096);
        assert_eq!(pt.total_bytes(), 4096);
    }

    #[test]
    fn transfer_estimated_bandwidth() {
        let topo = DeviceTopology::discover(2).unwrap();
        let pt = PeerTransfer::new(topo);
        let bw = pt.estimated_bandwidth(DeviceId(0), DeviceId(1));
        assert!(bw > 0.0);
    }

    #[test]
    fn transfer_clear_history() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        pt.record_transfer(DeviceId(0), DeviceId(1), 100, TransferDirection::DeviceToDevice);
        pt.clear_history();
        assert_eq!(pt.num_transfers(), 0);
        assert_eq!(pt.total_bytes(), 0);
    }

    // ── LoadBalancer ─────────────────────────────────────────────────

    #[test]
    fn lb_round_robin() {
        let topo = DeviceTopology::discover(3).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        let a0 = lb.assign(WorkItem::new("a", 1, 0)).unwrap();
        let a1 = lb.assign(WorkItem::new("b", 1, 0)).unwrap();
        let a2 = lb.assign(WorkItem::new("c", 1, 0)).unwrap();
        let a3 = lb.assign(WorkItem::new("d", 1, 0)).unwrap();
        assert_eq!(a0.device, DeviceId(0));
        assert_eq!(a1.device, DeviceId(1));
        assert_eq!(a2.device, DeviceId(2));
        assert_eq!(a3.device, DeviceId(0)); // wraps around
    }

    #[test]
    fn lb_least_loaded() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::LeastLoaded);
        lb.assign(WorkItem::new("heavy", 100, 0)).unwrap();
        let a2 = lb.assign(WorkItem::new("light", 1, 0)).unwrap();
        // Second item should go to the less loaded device.
        assert_eq!(a2.device, DeviceId(1));
    }

    #[test]
    fn lb_weighted_capacity() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::WeightedCapacity);
        let a = lb.assign(WorkItem::new("x", 1, 0)).unwrap();
        assert!(a.device == DeviceId(0) || a.device == DeviceId(1));
    }

    #[test]
    fn lb_memory_aware() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::MemoryAware);
        let a = lb.assign(WorkItem::new("x", 1, 1024)).unwrap();
        assert_eq!(lb.device_memory_used(a.device), 1024);
    }

    #[test]
    fn lb_memory_aware_oom() {
        let topo = DeviceTopology::discover(1).unwrap();
        let total = topo.device(DeviceId(0)).unwrap().total_memory;
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::MemoryAware);
        assert!(lb.assign(WorkItem::new("huge", 1, total + 1)).is_err());
    }

    #[test]
    fn lb_device_load_tracking() {
        let topo = DeviceTopology::discover(1).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        lb.assign(WorkItem::new("a", 5, 0)).unwrap();
        lb.assign(WorkItem::new("b", 3, 0)).unwrap();
        assert_eq!(lb.device_load(DeviceId(0)), 8);
    }

    #[test]
    fn lb_total_dispatched() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        for i in 0..7 {
            lb.assign(WorkItem::new(format!("w{i}"), 1, 0)).unwrap();
        }
        assert_eq!(lb.total_dispatched(), 7);
    }

    #[test]
    fn lb_assign_batch() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        let items: Vec<_> = (0..4).map(|i| WorkItem::new(format!("w{i}"), 1, 0)).collect();
        let assignments = lb.assign_batch(items).unwrap();
        assert_eq!(assignments.len(), 4);
    }

    #[test]
    fn lb_imbalance_ratio_balanced() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        lb.assign(WorkItem::new("a", 10, 0)).unwrap();
        lb.assign(WorkItem::new("b", 10, 0)).unwrap();
        assert!((lb.imbalance_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lb_imbalance_ratio_unbalanced() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::LeastLoaded);
        // All work to device 0 by force.
        *lb.device_load.get_mut(&DeviceId(0)).unwrap() = 100;
        *lb.device_load.get_mut(&DeviceId(1)).unwrap() = 0;
        // mean = 50, max = 100 → ratio = 2.0
        assert!((lb.imbalance_ratio() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn lb_imbalance_ratio_empty() {
        let topo = DeviceTopology::discover(2).unwrap();
        let lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        assert!((lb.imbalance_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lb_reset() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        lb.assign(WorkItem::new("a", 10, 1024)).unwrap();
        lb.reset();
        assert_eq!(lb.device_load(DeviceId(0)), 0);
        assert_eq!(lb.device_memory_used(DeviceId(0)), 0);
        assert_eq!(lb.total_dispatched(), 0);
    }

    #[test]
    fn lb_set_strategy() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::RoundRobin);
        assert_eq!(lb.strategy(), LoadBalancerStrategy::RoundRobin);
        lb.set_strategy(LoadBalancerStrategy::LeastLoaded);
        assert_eq!(lb.strategy(), LoadBalancerStrategy::LeastLoaded);
    }

    // ── Convenience functions ────────────────────────────────────────

    #[test]
    fn convenience_allreduce_sum() {
        let mut bufs = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        allreduce_sum(&mut bufs).unwrap();
        assert_eq!(bufs[0], vec![40.0, 60.0]);
    }

    #[test]
    fn convenience_create_context() {
        let ctx = create_multi_device_context(2).unwrap();
        assert_eq!(ctx.num_devices(), 2);
        assert!(ctx.is_synced());
    }

    // ── Integration / cross-component ────────────────────────────────

    #[test]
    fn end_to_end_split_transfer_reduce() {
        // Simulate: split tensor → transfer to devices → allreduce.
        let topo = DeviceTopology::discover(2).unwrap();
        let splitter = TensorSplitter::new(topo.clone(), SplitStrategy::Equal);
        let chunks = splitter.split(100, 4).unwrap();
        assert_eq!(chunks.len(), 2);

        // Simulate local computation producing per-device results.
        let mut bufs = vec![vec![1.0; 50], vec![2.0; 50]];
        allreduce_sum(&mut bufs).unwrap();
        assert_eq!(bufs[0][0], 3.0);
        assert_eq!(bufs[1][0], 3.0);
    }

    #[test]
    fn end_to_end_topology_ring_allreduce() {
        let topo = DeviceTopology::discover(4).unwrap();
        let ring = topo.optimal_ring_order();
        let cfg = AllReduceConfig { op: ReduceOp::Sum, ring_order: ring, chunk_size: 0 };
        let mut op = AllReduceOp::new(cfg, 4).unwrap();
        let mut bufs = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0], vec![4.0, 4.0]];
        op.execute(&mut bufs).unwrap();
        for buf in &bufs {
            assert_eq!(buf, &vec![10.0, 10.0]);
        }
    }

    #[test]
    fn end_to_end_load_balance_and_dispatch() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut ctx = MultiDeviceContext::new(topo.clone());
        let mut lb = LoadBalancer::new(topo, LoadBalancerStrategy::LeastLoaded);

        for i in 0..6 {
            let a = lb.assign(WorkItem::new(format!("k{i}"), 1, 0)).unwrap();
            ctx.dispatch_kernel(a.device).unwrap();
        }
        ctx.sync_all().unwrap();
        assert!(ctx.is_synced());
        assert_eq!(lb.total_dispatched(), 6);
    }

    #[test]
    fn end_to_end_peer_transfer_round_trip() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let original = vec![3.14, 2.72, 1.41];
        let mut buf_a = vec![0.0; 3];
        let mut buf_b = vec![0.0; 3];

        // GPU0 → GPU1.
        pt.transfer(DeviceId(0), DeviceId(1), &original, &mut buf_a).unwrap();
        // GPU1 → GPU0.
        pt.transfer(DeviceId(1), DeviceId(0), &buf_a, &mut buf_b).unwrap();
        assert_eq!(buf_b, original);
        assert_eq!(pt.num_transfers(), 2);
    }

    #[test]
    fn transfer_direction_variants() {
        // Ensure all variants are distinct.
        assert_ne!(TransferDirection::DeviceToDevice, TransferDirection::HostToDevice);
        assert_ne!(TransferDirection::HostToDevice, TransferDirection::DeviceToHost);
    }

    #[test]
    fn device_context_state_variants() {
        assert_ne!(DeviceContextState::Uninitialised, DeviceContextState::Ready);
        assert_ne!(DeviceContextState::Ready, DeviceContextState::Busy);
        assert_ne!(DeviceContextState::Busy, DeviceContextState::Error);
    }

    #[test]
    fn reduce_op_variants() {
        let ops = [ReduceOp::Sum, ReduceOp::Average, ReduceOp::Max, ReduceOp::Min];
        for (i, a) in ops.iter().enumerate() {
            for (j, b) in ops.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    // ── Additional coverage ──────────────────────────────────────────

    #[test]
    fn topology_link_bandwidth_matches_type() {
        let t = DeviceTopology::discover(3).unwrap();
        let link = t.link(DeviceId(0), DeviceId(1)).unwrap();
        assert_eq!(link.bandwidth_gbps, link.interconnect.bandwidth_gbps());
    }

    #[test]
    fn topology_link_latency_positive() {
        let t = DeviceTopology::discover(2).unwrap();
        let link = t.link(DeviceId(0), DeviceId(1)).unwrap();
        assert!(link.latency_us > 0.0);
    }

    #[test]
    fn splitter_large_element_count() {
        let topo = DeviceTopology::discover(8).unwrap();
        let s = TensorSplitter::new(topo, SplitStrategy::Equal);
        let chunks = s.split(1_000_000, 4).unwrap();
        let total: usize = chunks.iter().map(|c| c.length).sum();
        assert_eq!(total, 1_000_000);
        assert_eq!(chunks.len(), 8);
    }

    #[test]
    fn allreduce_large_buffer() {
        let n = 10_000;
        let mut bufs = vec![vec![1.0f32; n], vec![2.0f32; n]];
        allreduce_sum(&mut bufs).unwrap();
        for val in &bufs[0] {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn lb_all_strategies_work() {
        for strat in [
            LoadBalancerStrategy::RoundRobin,
            LoadBalancerStrategy::LeastLoaded,
            LoadBalancerStrategy::WeightedCapacity,
            LoadBalancerStrategy::MemoryAware,
        ] {
            let topo = DeviceTopology::discover(2).unwrap();
            let mut lb = LoadBalancer::new(topo, strat);
            let items: Vec<_> = (0..4).map(|i| WorkItem::new(format!("w{i}"), 1, 64)).collect();
            let assignments = lb.assign_batch(items).unwrap();
            assert_eq!(assignments.len(), 4);
        }
    }

    #[test]
    fn context_multiple_allocations() {
        let mut ctx = create_multi_device_context(2).unwrap();
        for _ in 0..10 {
            ctx.allocate(DeviceId(0), 100).unwrap();
        }
        assert_eq!(ctx.device_ctx(DeviceId(0)).unwrap().allocated_bytes, 1000);
    }

    #[test]
    fn context_id_unique() {
        let a = create_multi_device_context(1).unwrap();
        let b = create_multi_device_context(1).unwrap();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn transfer_id_unique() {
        let topo = DeviceTopology::discover(2).unwrap();
        let mut pt = PeerTransfer::new(topo);
        let r1 = pt.record_transfer(DeviceId(0), DeviceId(1), 1, TransferDirection::DeviceToDevice);
        let r2 = pt.record_transfer(DeviceId(0), DeviceId(1), 1, TransferDirection::DeviceToDevice);
        assert_ne!(r1.id, r2.id);
    }

    #[test]
    fn topology_device_none_returns_none() {
        let t = DeviceTopology::discover(2).unwrap();
        assert!(t.device(DeviceId(99)).is_none());
    }
}
