//! CUDA graph capture and replay for kernel sequence optimization.
//!
//! This module implements stream-based graph capture following the CUDA
//! `cudaStreamBeginCapture` / `cudaStreamEndCapture` paradigm:
//!
//! 1. [`GraphCapture::begin`] — start recording operations on a stream.
//! 2. Record kernel launches, memcpy, memset, host callbacks, events.
//! 3. [`GraphCapture::end`] — finalise the captured graph.
//! 4. [`GraphExec::launch`] — replay the captured sequence with minimal overhead.
//!
//! Key types:
//! - [`CaptureNodeKind`] — Kernel, Memcpy, Memset, Host, EventRecord.
//! - [`GraphExec`] — instantiated executable graph with launch/update support.
//! - [`GraphStats`] — profiling counters and timing data.
//!
//! All GPU dispatch is feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! CPU-only builds simulate capture and replay for testing.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bitnet_common::{KernelError, Result};

// ── Identifiers ──────────────────────────────────────────────────────

static NEXT_CAPTURE_NODE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_CAPTURE_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a capture node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaptureNodeId(u64);

impl CaptureNodeId {
    fn next() -> Self {
        Self(NEXT_CAPTURE_NODE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for CaptureNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cap-node-{}", self.0)
    }
}

/// Unique identifier for a captured graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaptureId(u64);

impl CaptureId {
    fn next() -> Self {
        Self(NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl fmt::Display for CaptureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "capture-{}", self.0)
    }
}

// ── Node kinds ───────────────────────────────────────────────────────

/// The kind of operation recorded during graph capture.
#[derive(Debug, Clone, PartialEq)]
pub enum CaptureNodeKind {
    /// GPU kernel launch.
    Kernel {
        /// Kernel function name.
        name: String,
        /// Grid dimensions (blocks).
        grid: [u32; 3],
        /// Block dimensions (threads per block).
        block: [u32; 3],
        /// Shared memory bytes.
        shared_mem_bytes: u32,
    },
    /// Device memory copy.
    Memcpy {
        /// Number of bytes to transfer.
        bytes: usize,
        /// Transfer direction.
        direction: MemcpyDirection,
    },
    /// Device memory set/fill.
    Memset {
        /// Number of bytes to set.
        bytes: usize,
        /// Fill value.
        value: u8,
    },
    /// Host-side callback.
    Host {
        /// Descriptive label for the callback.
        label: String,
    },
    /// Event record marker on the stream.
    EventRecord {
        /// Event name for correlation.
        event_name: String,
    },
}

/// Direction of a memory copy operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyDirection {
    /// Host to device.
    HostToDevice,
    /// Device to host.
    DeviceToHost,
    /// Device to device.
    DeviceToDevice,
}

impl fmt::Display for MemcpyDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H2D"),
            Self::DeviceToHost => write!(f, "D2H"),
            Self::DeviceToDevice => write!(f, "D2D"),
        }
    }
}

// ── CaptureNode ──────────────────────────────────────────────────────

/// A recorded operation in the capture sequence.
#[derive(Debug, Clone)]
pub struct CaptureNode {
    /// Unique node identifier.
    pub id: CaptureNodeId,
    /// Operation kind.
    pub kind: CaptureNodeKind,
    /// Arbitrary parameters (kernel arguments, tuning knobs).
    pub params: HashMap<String, f64>,
    /// Stream index for multi-stream captures.
    pub stream: u32,
}

impl CaptureNode {
    /// Create a kernel launch node.
    pub fn kernel(name: &str, grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            id: CaptureNodeId::next(),
            kind: CaptureNodeKind::Kernel {
                name: name.to_string(),
                grid,
                block,
                shared_mem_bytes: 0,
            },
            params: HashMap::new(),
            stream: 0,
        }
    }

    /// Create a kernel node with shared memory.
    pub fn kernel_with_shared(
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
    ) -> Self {
        Self {
            id: CaptureNodeId::next(),
            kind: CaptureNodeKind::Kernel { name: name.to_string(), grid, block, shared_mem_bytes },
            params: HashMap::new(),
            stream: 0,
        }
    }

    /// Create a memcpy node.
    pub fn memcpy(bytes: usize, direction: MemcpyDirection) -> Self {
        Self {
            id: CaptureNodeId::next(),
            kind: CaptureNodeKind::Memcpy { bytes, direction },
            params: HashMap::new(),
            stream: 0,
        }
    }

    /// Create a memset node.
    pub fn memset(bytes: usize, value: u8) -> Self {
        Self {
            id: CaptureNodeId::next(),
            kind: CaptureNodeKind::Memset { bytes, value },
            params: HashMap::new(),
            stream: 0,
        }
    }

    /// Create a host callback node.
    pub fn host(label: &str) -> Self {
        Self {
            id: CaptureNodeId::next(),
            kind: CaptureNodeKind::Host { label: label.to_string() },
            params: HashMap::new(),
            stream: 0,
        }
    }

    /// Create an event record node.
    pub fn event_record(event_name: &str) -> Self {
        Self {
            id: CaptureNodeId::next(),
            kind: CaptureNodeKind::EventRecord { event_name: event_name.to_string() },
            params: HashMap::new(),
            stream: 0,
        }
    }

    /// Set a parameter value.
    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// Assign to a specific stream.
    pub fn on_stream(mut self, stream: u32) -> Self {
        self.stream = stream;
        self
    }

    /// Whether this node is a kernel launch.
    pub fn is_kernel(&self) -> bool {
        matches!(self.kind, CaptureNodeKind::Kernel { .. })
    }

    /// Whether this node is a memory operation (memcpy or memset).
    pub fn is_memory_op(&self) -> bool {
        matches!(self.kind, CaptureNodeKind::Memcpy { .. } | CaptureNodeKind::Memset { .. })
    }

    /// Estimated execution cost in microseconds (heuristic).
    pub fn estimated_cost_us(&self) -> f64 {
        match &self.kind {
            CaptureNodeKind::Kernel { grid, block, .. } => {
                let threads = grid[0] as f64
                    * grid[1] as f64
                    * grid[2] as f64
                    * block[0] as f64
                    * block[1] as f64
                    * block[2] as f64;
                1.0 + threads * 0.001
            }
            CaptureNodeKind::Memcpy { bytes, .. } | CaptureNodeKind::Memset { bytes, .. } => {
                // ~400 GB/s bandwidth estimate
                0.5 + *bytes as f64 / (400.0 * 1e9 / 1e6)
            }
            CaptureNodeKind::Host { .. } => 5.0,
            CaptureNodeKind::EventRecord { .. } => 0.1,
        }
    }

    /// Kernel name if this is a kernel node.
    pub fn kernel_name(&self) -> Option<&str> {
        match &self.kind {
            CaptureNodeKind::Kernel { name, .. } => Some(name),
            _ => None,
        }
    }
}

// ── Capture state machine ────────────────────────────────────────────

/// State of stream-based graph capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureStatus {
    /// Not capturing — initial state.
    Idle,
    /// Actively recording operations.
    Recording,
    /// Capture completed successfully.
    Complete,
    /// Capture was invalidated (e.g. error during recording).
    Invalidated,
}

// ── CudaGraph (captured) ─────────────────────────────────────────────

/// A captured CUDA graph representing a recorded sequence of GPU operations.
///
/// On CPU-only builds the graph stores the recorded sequence and simulates
/// replay in recorded order.
#[derive(Debug, Clone)]
pub struct CudaGraph {
    /// Unique capture identifier.
    pub id: CaptureId,
    /// Recorded nodes in capture order.
    nodes: Vec<CaptureNode>,
    /// Optional label for debugging.
    label: String,
    /// When the capture completed.
    captured_at: Option<Instant>,
}

impl CudaGraph {
    /// Number of recorded nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Slice of all recorded nodes.
    pub fn nodes(&self) -> &[CaptureNode] {
        &self.nodes
    }

    /// Debug label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Whether capture has completed.
    pub fn is_captured(&self) -> bool {
        self.captured_at.is_some()
    }

    /// Count kernel nodes.
    pub fn kernel_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_kernel()).count()
    }

    /// Count memory operation nodes.
    pub fn memory_op_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_memory_op()).count()
    }

    /// Count event record nodes.
    pub fn event_count(&self) -> usize {
        self.nodes.iter().filter(|n| matches!(n.kind, CaptureNodeKind::EventRecord { .. })).count()
    }

    /// Count host callback nodes.
    pub fn host_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| matches!(n.kind, CaptureNodeKind::Host { .. })).count()
    }

    /// Total estimated cost in microseconds.
    pub fn total_estimated_cost_us(&self) -> f64 {
        self.nodes.iter().map(|n| n.estimated_cost_us()).sum()
    }

    /// Find a node by id.
    pub fn find_node(&self, id: CaptureNodeId) -> Option<&CaptureNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Distinct stream indices used.
    pub fn stream_count(&self) -> usize {
        let streams: std::collections::HashSet<u32> = self.nodes.iter().map(|n| n.stream).collect();
        streams.len()
    }
}

// ── GraphCapture (builder / state machine) ───────────────────────────

/// Stream-based graph capture builder.
///
/// Usage:
/// ```ignore
/// let mut cap = GraphCapture::begin("my_graph")?;
/// cap.record_kernel("matmul", [64, 1, 1], [256, 1, 1])?;
/// cap.record_memcpy(4096, MemcpyDirection::DeviceToDevice)?;
/// let graph = cap.end()?;
/// ```
pub struct GraphCapture {
    label: String,
    nodes: Vec<CaptureNode>,
    status: CaptureStatus,
    started_at: Instant,
}

impl GraphCapture {
    /// Begin graph capture on a stream.
    pub fn begin(label: &str) -> Result<Self> {
        if label.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "capture label must not be empty".into(),
            }
            .into());
        }
        Ok(Self {
            label: label.to_string(),
            nodes: Vec::new(),
            status: CaptureStatus::Recording,
            started_at: Instant::now(),
        })
    }

    /// Current capture status.
    pub fn status(&self) -> CaptureStatus {
        self.status
    }

    /// Number of nodes recorded so far.
    pub fn recorded_count(&self) -> usize {
        self.nodes.len()
    }

    /// Record a kernel launch.
    pub fn record_kernel(
        &mut self,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<CaptureNodeId> {
        self.check_recording()?;
        Self::validate_dims(grid, block)?;
        let node = CaptureNode::kernel(name, grid, block);
        let id = node.id;
        self.nodes.push(node);
        Ok(id)
    }

    /// Record a kernel launch with shared memory.
    pub fn record_kernel_shared(
        &mut self,
        name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
    ) -> Result<CaptureNodeId> {
        self.check_recording()?;
        Self::validate_dims(grid, block)?;
        let node = CaptureNode::kernel_with_shared(name, grid, block, shared_mem_bytes);
        let id = node.id;
        self.nodes.push(node);
        Ok(id)
    }

    /// Record a memory copy.
    pub fn record_memcpy(
        &mut self,
        bytes: usize,
        direction: MemcpyDirection,
    ) -> Result<CaptureNodeId> {
        self.check_recording()?;
        if bytes == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "memcpy bytes must be > 0".into(),
            }
            .into());
        }
        let node = CaptureNode::memcpy(bytes, direction);
        let id = node.id;
        self.nodes.push(node);
        Ok(id)
    }

    /// Record a memory set operation.
    pub fn record_memset(&mut self, bytes: usize, value: u8) -> Result<CaptureNodeId> {
        self.check_recording()?;
        if bytes == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "memset bytes must be > 0".into(),
            }
            .into());
        }
        let node = CaptureNode::memset(bytes, value);
        let id = node.id;
        self.nodes.push(node);
        Ok(id)
    }

    /// Record a host-side callback.
    pub fn record_host(&mut self, label: &str) -> Result<CaptureNodeId> {
        self.check_recording()?;
        let node = CaptureNode::host(label);
        let id = node.id;
        self.nodes.push(node);
        Ok(id)
    }

    /// Record an event on the stream.
    pub fn record_event(&mut self, event_name: &str) -> Result<CaptureNodeId> {
        self.check_recording()?;
        let node = CaptureNode::event_record(event_name);
        let id = node.id;
        self.nodes.push(node);
        Ok(id)
    }

    /// Finalise capture and produce a [`CudaGraph`].
    pub fn end(mut self) -> Result<CudaGraph> {
        if self.status != CaptureStatus::Recording {
            return Err(KernelError::InvalidArguments {
                reason: format!("capture is not recording (status: {:?})", self.status),
            }
            .into());
        }
        if self.nodes.is_empty() {
            return Err(KernelError::InvalidArguments {
                reason: "cannot end capture with no recorded operations".into(),
            }
            .into());
        }
        self.status = CaptureStatus::Complete;
        Ok(CudaGraph {
            id: CaptureId::next(),
            nodes: self.nodes,
            label: self.label,
            captured_at: Some(Instant::now()),
        })
    }

    /// Invalidate the capture (e.g. on error).
    pub fn invalidate(&mut self) {
        self.status = CaptureStatus::Invalidated;
    }

    /// Duration since capture began.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    fn check_recording(&self) -> Result<()> {
        if self.status != CaptureStatus::Recording {
            return Err(KernelError::InvalidArguments {
                reason: format!("capture is not recording (status: {:?})", self.status),
            }
            .into());
        }
        Ok(())
    }

    fn validate_dims(grid: [u32; 3], block: [u32; 3]) -> Result<()> {
        if grid.contains(&0) {
            return Err(KernelError::InvalidArguments {
                reason: "grid dimensions must be non-zero".into(),
            }
            .into());
        }
        if block.contains(&0) {
            return Err(KernelError::InvalidArguments {
                reason: "block dimensions must be non-zero".into(),
            }
            .into());
        }
        let threads_per_block = block[0] as u64 * block[1] as u64 * block[2] as u64;
        if threads_per_block > 1024 {
            return Err(KernelError::InvalidArguments {
                reason: format!("threads per block ({threads_per_block}) exceeds maximum (1024)"),
            }
            .into());
        }
        Ok(())
    }
}

// ── GraphExec ────────────────────────────────────────────────────────

/// An instantiated executable graph ready for replay.
///
/// On CPU-only builds, `launch` simulates execution by walking the
/// recorded node sequence.
#[derive(Debug)]
pub struct GraphExec {
    /// The captured graph this executor was instantiated from.
    graph: CudaGraph,
    /// Number of times this executor has been launched.
    launch_count: u64,
    /// Cumulative wall-clock time across all launches.
    total_launch_time: Duration,
}

impl GraphExec {
    /// Instantiate a captured graph for execution.
    pub fn instantiate(graph: CudaGraph) -> Result<Self> {
        if !graph.is_captured() {
            return Err(KernelError::InvalidArguments {
                reason: "graph has not been captured".into(),
            }
            .into());
        }
        if graph.node_count() == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "cannot instantiate graph with no nodes".into(),
            }
            .into());
        }
        Ok(Self { graph, launch_count: 0, total_launch_time: Duration::ZERO })
    }

    /// Launch (replay) the captured graph.
    pub fn launch(&mut self) -> Result<LaunchResult> {
        let start = Instant::now();
        let mut nodes_executed = 0usize;
        let mut estimated_gpu_us = 0.0f64;

        for node in &self.graph.nodes {
            nodes_executed += 1;
            estimated_gpu_us += node.estimated_cost_us();
        }

        let wall_time = start.elapsed();
        self.launch_count += 1;
        self.total_launch_time += wall_time;

        Ok(LaunchResult { capture_id: self.graph.id, nodes_executed, wall_time, estimated_gpu_us })
    }

    /// Number of launches.
    pub fn launch_count(&self) -> u64 {
        self.launch_count
    }

    /// Total wall-clock time across all launches.
    pub fn total_launch_time(&self) -> Duration {
        self.total_launch_time
    }

    /// Average launch time.
    pub fn avg_launch_time(&self) -> Duration {
        if self.launch_count == 0 {
            return Duration::ZERO;
        }
        self.total_launch_time / self.launch_count as u32
    }

    /// Access the underlying captured graph.
    pub fn graph(&self) -> &CudaGraph {
        &self.graph
    }

    /// Update kernel parameters without recapture.
    ///
    /// Finds all kernel nodes matching `kernel_name` and applies the
    /// given parameter updates. Returns the number of nodes updated.
    pub fn update_kernel_params(
        &mut self,
        kernel_name: &str,
        params: &HashMap<String, f64>,
    ) -> Result<usize> {
        let mut updated = 0usize;
        for node in &mut self.graph.nodes {
            if let CaptureNodeKind::Kernel { name, .. } = &node.kind
                && name == kernel_name
            {
                for (k, v) in params {
                    node.params.insert(k.clone(), *v);
                }
                updated += 1;
            }
        }
        if updated == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!("no kernel node named '{kernel_name}' found"),
            }
            .into());
        }
        Ok(updated)
    }

    /// Swap a kernel node to a different kernel function without recapture.
    ///
    /// Finds the first kernel node matching `old_name` and replaces its
    /// name with `new_name`, optionally updating grid/block dims.
    pub fn swap_kernel(
        &mut self,
        old_name: &str,
        new_name: &str,
        new_grid: Option<[u32; 3]>,
        new_block: Option<[u32; 3]>,
    ) -> Result<()> {
        for node in &mut self.graph.nodes {
            if let CaptureNodeKind::Kernel { name, grid, block, .. } = &mut node.kind
                && name == old_name
            {
                *name = new_name.to_string();
                if let Some(g) = new_grid {
                    *grid = g;
                }
                if let Some(b) = new_block {
                    *block = b;
                }
                return Ok(());
            }
        }
        Err(KernelError::InvalidArguments {
            reason: format!("no kernel node named '{old_name}' found"),
        }
        .into())
    }
}

/// Result of a graph launch.
#[derive(Debug, Clone)]
pub struct LaunchResult {
    /// Capture identifier.
    pub capture_id: CaptureId,
    /// Number of nodes executed.
    pub nodes_executed: usize,
    /// Wall-clock time for the simulated launch.
    pub wall_time: Duration,
    /// Estimated GPU time in microseconds.
    pub estimated_gpu_us: f64,
}

// ── GraphStats ───────────────────────────────────────────────────────

/// Profiling statistics for graph launches.
pub struct GraphStats {
    samples: Vec<LaunchResult>,
    max_samples: usize,
}

impl GraphStats {
    /// Create a new stats collector with a sample budget.
    pub fn new(max_samples: usize) -> Self {
        Self { samples: Vec::new(), max_samples }
    }

    /// Record a launch result.
    pub fn record(&mut self, result: LaunchResult) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(result);
    }

    /// Number of recorded samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// All samples.
    pub fn samples(&self) -> &[LaunchResult] {
        &self.samples
    }

    /// Average wall-clock time.
    pub fn avg_wall_time(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.samples.iter().map(|s| s.wall_time).sum();
        total / self.samples.len() as u32
    }

    /// Average estimated GPU time in µs.
    pub fn avg_estimated_gpu_us(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total: f64 = self.samples.iter().map(|s| s.estimated_gpu_us).sum();
        total / self.samples.len() as f64
    }

    /// Minimum wall-clock time.
    pub fn min_wall_time(&self) -> Duration {
        self.samples.iter().map(|s| s.wall_time).min().unwrap_or(Duration::ZERO)
    }

    /// Maximum wall-clock time.
    pub fn max_wall_time(&self) -> Duration {
        self.samples.iter().map(|s| s.wall_time).max().unwrap_or(Duration::ZERO)
    }

    /// Total nodes executed across all samples.
    pub fn total_nodes_executed(&self) -> usize {
        self.samples.iter().map(|s| s.nodes_executed).sum()
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CaptureNodeId / CaptureId ────────────────────────────────

    #[test]
    fn capture_node_id_uniqueness() {
        let a = CaptureNodeId::next();
        let b = CaptureNodeId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn capture_id_uniqueness() {
        let a = CaptureId::next();
        let b = CaptureId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn capture_node_id_display() {
        let id = CaptureNodeId(42);
        assert_eq!(format!("{id}"), "cap-node-42");
    }

    #[test]
    fn capture_id_display() {
        let id = CaptureId(7);
        assert_eq!(format!("{id}"), "capture-7");
    }

    // ── CaptureNode construction ─────────────────────────────────

    #[test]
    fn kernel_node_basic() {
        let n = CaptureNode::kernel("matmul", [8, 1, 1], [256, 1, 1]);
        assert!(n.is_kernel());
        assert!(!n.is_memory_op());
        assert_eq!(n.kernel_name(), Some("matmul"));
        assert_eq!(n.stream, 0);
    }

    #[test]
    fn kernel_with_shared_mem() {
        let n = CaptureNode::kernel_with_shared("attn", [4, 1, 1], [128, 1, 1], 8192);
        if let CaptureNodeKind::Kernel { shared_mem_bytes, .. } = &n.kind {
            assert_eq!(*shared_mem_bytes, 8192);
        } else {
            panic!("expected Kernel");
        }
    }

    #[test]
    fn memcpy_node() {
        let n = CaptureNode::memcpy(4096, MemcpyDirection::HostToDevice);
        assert!(n.is_memory_op());
        assert!(!n.is_kernel());
        assert_eq!(n.kernel_name(), None);
    }

    #[test]
    fn memset_node() {
        let n = CaptureNode::memset(1024, 0);
        assert!(n.is_memory_op());
        if let CaptureNodeKind::Memset { bytes, value } = &n.kind {
            assert_eq!(*bytes, 1024);
            assert_eq!(*value, 0);
        } else {
            panic!("expected Memset");
        }
    }

    #[test]
    fn host_callback_node() {
        let n = CaptureNode::host("my_callback");
        assert!(!n.is_kernel());
        assert!(!n.is_memory_op());
        if let CaptureNodeKind::Host { label } = &n.kind {
            assert_eq!(label, "my_callback");
        } else {
            panic!("expected Host");
        }
    }

    #[test]
    fn event_record_node() {
        let n = CaptureNode::event_record("sync_point");
        if let CaptureNodeKind::EventRecord { event_name } = &n.kind {
            assert_eq!(event_name, "sync_point");
        } else {
            panic!("expected EventRecord");
        }
    }

    #[test]
    fn node_with_param() {
        let n = CaptureNode::kernel("k", [1, 1, 1], [1, 1, 1]).with_param("alpha", 2.0);
        assert!((n.params["alpha"] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn node_on_stream() {
        let n = CaptureNode::kernel("k", [1, 1, 1], [1, 1, 1]).on_stream(3);
        assert_eq!(n.stream, 3);
    }

    #[test]
    fn estimated_cost_kernel_positive() {
        let n = CaptureNode::kernel("k", [4, 1, 1], [256, 1, 1]);
        assert!(n.estimated_cost_us() > 0.0);
    }

    #[test]
    fn estimated_cost_memcpy_positive() {
        let n = CaptureNode::memcpy(1_000_000, MemcpyDirection::DeviceToDevice);
        assert!(n.estimated_cost_us() > 0.0);
    }

    #[test]
    fn estimated_cost_event_small() {
        let n = CaptureNode::event_record("ev");
        assert!(n.estimated_cost_us() < 1.0);
    }

    // ── MemcpyDirection display ──────────────────────────────────

    #[test]
    fn memcpy_direction_display() {
        assert_eq!(format!("{}", MemcpyDirection::HostToDevice), "H2D");
        assert_eq!(format!("{}", MemcpyDirection::DeviceToHost), "D2H");
        assert_eq!(format!("{}", MemcpyDirection::DeviceToDevice), "D2D");
    }

    // ── GraphCapture lifecycle ───────────────────────────────────

    #[test]
    fn begin_capture() {
        let cap = GraphCapture::begin("test").unwrap();
        assert_eq!(cap.status(), CaptureStatus::Recording);
        assert_eq!(cap.recorded_count(), 0);
    }

    #[test]
    fn begin_capture_empty_label_fails() {
        assert!(GraphCapture::begin("").is_err());
    }

    #[test]
    fn record_and_end() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("matmul", [1, 1, 1], [128, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        assert_eq!(graph.node_count(), 1);
        assert!(graph.is_captured());
        assert_eq!(graph.label(), "test");
    }

    #[test]
    fn end_empty_capture_fails() {
        let cap = GraphCapture::begin("test").unwrap();
        assert!(cap.end().is_err());
    }

    #[test]
    fn record_after_end_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [1, 1, 1]).unwrap();
        // Move cap into end → consumed. Cannot record after.
        // This is enforced by ownership (end takes self).
        // Instead test invalidation:
        let mut cap2 = GraphCapture::begin("test2").unwrap();
        cap2.invalidate();
        assert!(cap2.record_kernel("k", [1, 1, 1], [1, 1, 1]).is_err());
    }

    #[test]
    fn invalidated_capture_cannot_end() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [1, 1, 1]).unwrap();
        cap.invalidate();
        assert_eq!(cap.status(), CaptureStatus::Invalidated);
        assert!(cap.end().is_err());
    }

    #[test]
    fn record_all_node_types() {
        let mut cap = GraphCapture::begin("mixed").unwrap();
        cap.record_kernel("k1", [1, 1, 1], [32, 1, 1]).unwrap();
        cap.record_memcpy(4096, MemcpyDirection::HostToDevice).unwrap();
        cap.record_memset(1024, 0xFF).unwrap();
        cap.record_host("callback").unwrap();
        cap.record_event("sync").unwrap();
        let graph = cap.end().unwrap();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.kernel_count(), 1);
        assert_eq!(graph.memory_op_count(), 2);
        assert_eq!(graph.event_count(), 1);
        assert_eq!(graph.host_node_count(), 1);
    }

    #[test]
    fn record_kernel_zero_grid_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        assert!(cap.record_kernel("k", [0, 1, 1], [1, 1, 1]).is_err());
    }

    #[test]
    fn record_kernel_zero_block_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        assert!(cap.record_kernel("k", [1, 1, 1], [0, 1, 1]).is_err());
    }

    #[test]
    fn record_kernel_exceeds_thread_limit_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        // 1025 threads per block exceeds CUDA max of 1024
        assert!(cap.record_kernel("k", [1, 1, 1], [1025, 1, 1]).is_err());
    }

    #[test]
    fn record_memcpy_zero_bytes_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        assert!(cap.record_memcpy(0, MemcpyDirection::HostToDevice).is_err());
    }

    #[test]
    fn record_memset_zero_bytes_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        assert!(cap.record_memset(0, 0).is_err());
    }

    #[test]
    fn record_kernel_shared_mem() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel_shared("attn", [4, 1, 1], [128, 1, 1], 16384).unwrap();
        let graph = cap.end().unwrap();
        assert_eq!(graph.kernel_count(), 1);
    }

    #[test]
    fn capture_elapsed_positive() {
        let cap = GraphCapture::begin("test").unwrap();
        assert!(cap.elapsed() >= Duration::ZERO);
    }

    // ── CudaGraph queries ────────────────────────────────────────

    #[test]
    fn graph_find_node() {
        let mut cap = GraphCapture::begin("test").unwrap();
        let id = cap.record_kernel("k", [1, 1, 1], [32, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        assert!(graph.find_node(id).is_some());
        assert!(graph.find_node(CaptureNodeId(999_999)).is_none());
    }

    #[test]
    fn graph_stream_count() {
        let mut cap = GraphCapture::begin("test").unwrap();
        // Record nodes on different streams by building manually
        let mut n1 = CaptureNode::kernel("a", [1, 1, 1], [1, 1, 1]);
        n1.stream = 0;
        let mut n2 = CaptureNode::kernel("b", [1, 1, 1], [1, 1, 1]);
        n2.stream = 1;
        cap.nodes.push(n1);
        cap.nodes.push(n2);
        let graph = cap.end().unwrap();
        assert_eq!(graph.stream_count(), 2);
    }

    #[test]
    fn graph_total_estimated_cost() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [4, 1, 1], [256, 1, 1]).unwrap();
        cap.record_memcpy(1_000_000, MemcpyDirection::DeviceToDevice).unwrap();
        let graph = cap.end().unwrap();
        assert!(graph.total_estimated_cost_us() > 0.0);
    }

    // ── GraphExec ────────────────────────────────────────────────

    #[test]
    fn instantiate_and_launch() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [32, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();
        let res = exec.launch().unwrap();
        assert_eq!(res.nodes_executed, 1);
        assert_eq!(exec.launch_count(), 1);
    }

    #[test]
    fn multiple_launches() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [32, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();
        exec.launch().unwrap();
        exec.launch().unwrap();
        exec.launch().unwrap();
        assert_eq!(exec.launch_count(), 3);
        assert!(exec.total_launch_time() >= Duration::ZERO);
        assert!(exec.avg_launch_time() >= Duration::ZERO);
    }

    #[test]
    fn avg_launch_time_zero_when_no_launches() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [1, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let exec = GraphExec::instantiate(graph).unwrap();
        assert_eq!(exec.avg_launch_time(), Duration::ZERO);
    }

    #[test]
    fn update_kernel_params_succeeds() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("matmul", [4, 1, 1], [256, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();

        let mut params = HashMap::new();
        params.insert("alpha".to_string(), 2.0);
        let count = exec.update_kernel_params("matmul", &params).unwrap();
        assert_eq!(count, 1);
        assert!((exec.graph().nodes()[0].params["alpha"] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn update_kernel_params_no_match_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [1, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();
        let params = HashMap::new();
        assert!(exec.update_kernel_params("nonexistent", &params).is_err());
    }

    #[test]
    fn swap_kernel_succeeds() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("old_gemm", [4, 1, 1], [256, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();

        exec.swap_kernel("old_gemm", "new_gemm", Some([8, 1, 1]), None).unwrap();
        let node = &exec.graph().nodes()[0];
        assert_eq!(node.kernel_name(), Some("new_gemm"));
        if let CaptureNodeKind::Kernel { grid, .. } = &node.kind {
            assert_eq!(grid[0], 8);
        }
    }

    #[test]
    fn swap_kernel_not_found_fails() {
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [1, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();
        assert!(exec.swap_kernel("missing", "new", None, None).is_err());
    }

    #[test]
    fn exec_graph_accessor() {
        let mut cap = GraphCapture::begin("label").unwrap();
        cap.record_kernel("k", [1, 1, 1], [1, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let exec = GraphExec::instantiate(graph).unwrap();
        assert_eq!(exec.graph().label(), "label");
    }

    // ── GraphStats ───────────────────────────────────────────────

    #[test]
    fn stats_basic() {
        let stats = GraphStats::new(100);
        assert_eq!(stats.sample_count(), 0);
        assert_eq!(stats.avg_wall_time(), Duration::ZERO);
        assert_eq!(stats.avg_estimated_gpu_us(), 0.0);
    }

    #[test]
    fn stats_record_and_query() {
        let mut stats = GraphStats::new(100);
        let mut cap = GraphCapture::begin("test").unwrap();
        cap.record_kernel("k", [1, 1, 1], [32, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        let mut exec = GraphExec::instantiate(graph).unwrap();
        let res = exec.launch().unwrap();
        stats.record(res);
        assert_eq!(stats.sample_count(), 1);
        assert!(stats.avg_estimated_gpu_us() > 0.0);
    }

    #[test]
    fn stats_evicts_oldest() {
        let mut stats = GraphStats::new(2);
        for _ in 0..3 {
            stats.record(LaunchResult {
                capture_id: CaptureId(1),
                nodes_executed: 1,
                wall_time: Duration::from_micros(100),
                estimated_gpu_us: 10.0,
            });
        }
        assert_eq!(stats.sample_count(), 2);
    }

    #[test]
    fn stats_min_max_wall_time() {
        let mut stats = GraphStats::new(100);
        stats.record(LaunchResult {
            capture_id: CaptureId(1),
            nodes_executed: 1,
            wall_time: Duration::from_micros(50),
            estimated_gpu_us: 5.0,
        });
        stats.record(LaunchResult {
            capture_id: CaptureId(1),
            nodes_executed: 1,
            wall_time: Duration::from_micros(150),
            estimated_gpu_us: 15.0,
        });
        assert!(stats.min_wall_time() <= stats.max_wall_time());
        assert_eq!(stats.min_wall_time(), Duration::from_micros(50));
        assert_eq!(stats.max_wall_time(), Duration::from_micros(150));
    }

    #[test]
    fn stats_total_nodes_executed() {
        let mut stats = GraphStats::new(100);
        stats.record(LaunchResult {
            capture_id: CaptureId(1),
            nodes_executed: 5,
            wall_time: Duration::ZERO,
            estimated_gpu_us: 0.0,
        });
        stats.record(LaunchResult {
            capture_id: CaptureId(1),
            nodes_executed: 3,
            wall_time: Duration::ZERO,
            estimated_gpu_us: 0.0,
        });
        assert_eq!(stats.total_nodes_executed(), 8);
    }

    #[test]
    fn stats_clear() {
        let mut stats = GraphStats::new(100);
        stats.record(LaunchResult {
            capture_id: CaptureId(1),
            nodes_executed: 1,
            wall_time: Duration::ZERO,
            estimated_gpu_us: 0.0,
        });
        stats.clear();
        assert_eq!(stats.sample_count(), 0);
    }

    // ── Integration: capture → exec → stats ──────────────────────

    #[test]
    fn full_capture_exec_stats_workflow() {
        // 1. Capture
        let mut cap = GraphCapture::begin("inference_step").unwrap();
        cap.record_kernel("rmsnorm", [8, 1, 1], [256, 1, 1]).unwrap();
        cap.record_kernel("qkv_proj", [16, 1, 1], [256, 1, 1]).unwrap();
        cap.record_kernel("attention", [8, 4, 1], [128, 1, 1]).unwrap();
        cap.record_memcpy(2048, MemcpyDirection::DeviceToDevice).unwrap();
        cap.record_event("attn_done").unwrap();
        cap.record_kernel("ffn", [16, 1, 1], [256, 1, 1]).unwrap();
        let graph = cap.end().unwrap();
        assert_eq!(graph.kernel_count(), 4);

        // 2. Instantiate and launch
        let mut exec = GraphExec::instantiate(graph).unwrap();
        let mut stats = GraphStats::new(10);
        for _ in 0..5 {
            let res = exec.launch().unwrap();
            stats.record(res);
        }

        // 3. Check stats
        assert_eq!(exec.launch_count(), 5);
        assert_eq!(stats.sample_count(), 5);
        assert!(stats.avg_estimated_gpu_us() > 0.0);

        // 4. Update params without recapture
        let mut params = HashMap::new();
        params.insert("eps".to_string(), 1e-5);
        exec.update_kernel_params("rmsnorm", &params).unwrap();

        // 5. Launch again after update
        let res = exec.launch().unwrap();
        assert!(res.nodes_executed > 0);
    }
}
