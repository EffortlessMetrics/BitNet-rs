//! Parallel communication primitives for distributed GPU inference.
//!
//! Provides collective operations (all-reduce, all-gather, reduce-scatter,
//! broadcast), communication topologies (ring, tree), double-buffered
//! communication, profiling, and a unified engine for multi-device inference.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Error ──────────────────────────────────────────────────────────────

/// Errors from parallel communication operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommError {
    /// Worker id is out of range for the current configuration.
    InvalidWorker(usize),
    /// Data length mismatch between workers.
    SizeMismatch { expected: usize, actual: usize },
    /// The buffer has not been allocated.
    BufferNotReady,
    /// The topology has no workers.
    EmptyTopology,
    /// Operation not supported for the current backend.
    UnsupportedOperation(String),
}

impl fmt::Display for CommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorker(id) => write!(f, "invalid worker id: {id}"),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "size mismatch: expected {expected}, got {actual}")
            }
            Self::BufferNotReady => write!(f, "communication buffer not ready"),
            Self::EmptyTopology => write!(f, "topology has no workers"),
            Self::UnsupportedOperation(op) => {
                write!(f, "unsupported operation: {op}")
            }
        }
    }
}

// ── ReduceOp ───────────────────────────────────────────────────────────

/// Reduction operation applied element-wise across workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise mean (sum / num_workers).
    Mean,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sum => write!(f, "Sum"),
            Self::Mean => write!(f, "Mean"),
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
        }
    }
}

// ── CommBackend ────────────────────────────────────────────────────────

/// Communication backend used for data transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommBackend {
    /// Simulated in-process communication for testing.
    Simulated,
    /// Shared-memory transport for single-node multi-GPU.
    SharedMemory,
    /// Network transport for multi-node clusters.
    Network,
}

impl fmt::Display for CommBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Simulated => write!(f, "Simulated"),
            Self::SharedMemory => write!(f, "SharedMemory"),
            Self::Network => write!(f, "Network"),
        }
    }
}

impl Default for CommBackend {
    fn default() -> Self {
        Self::Simulated
    }
}

// ── TopologyKind ───────────────────────────────────────────────────────

/// Topology strategy for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopologyKind {
    /// Ring-based: each worker communicates with its left/right neighbours.
    Ring,
    /// Tree-based: hierarchical reduction/broadcast.
    Tree,
}

impl fmt::Display for TopologyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ring => write!(f, "Ring"),
            Self::Tree => write!(f, "Tree"),
        }
    }
}

impl Default for TopologyKind {
    fn default() -> Self {
        Self::Ring
    }
}

// ── CommConfig ─────────────────────────────────────────────────────────

/// Configuration for the parallel communication engine.
///
/// Specifies topology, backend, buffer sizes, and whether asynchronous
/// overlap of communication and computation is enabled.
#[derive(Debug, Clone)]
pub struct CommConfig {
    /// Number of workers (devices / processes) participating.
    num_workers: usize,
    /// Topology strategy.
    topology: TopologyKind,
    /// Transport backend.
    backend: CommBackend,
    /// Per-worker send buffer capacity in elements.
    send_buffer_capacity: usize,
    /// Per-worker receive buffer capacity in elements.
    recv_buffer_capacity: usize,
    /// Enable asynchronous communication/computation overlap.
    async_mode: bool,
}

impl CommConfig {
    /// Create a new configuration with defaults for the given worker count.
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            topology: TopologyKind::default(),
            backend: CommBackend::default(),
            send_buffer_capacity: 4096,
            recv_buffer_capacity: 4096,
            async_mode: false,
        }
    }

    /// Set the topology strategy.
    pub fn with_topology(mut self, topology: TopologyKind) -> Self {
        self.topology = topology;
        self
    }

    /// Set the communication backend.
    pub fn with_backend(mut self, backend: CommBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the send buffer capacity.
    pub fn with_send_buffer_capacity(mut self, cap: usize) -> Self {
        self.send_buffer_capacity = cap;
        self
    }

    /// Set the receive buffer capacity.
    pub fn with_recv_buffer_capacity(mut self, cap: usize) -> Self {
        self.recv_buffer_capacity = cap;
        self
    }

    /// Enable or disable async overlap mode.
    pub fn with_async_mode(mut self, enabled: bool) -> Self {
        self.async_mode = enabled;
        self
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
    pub fn topology(&self) -> TopologyKind {
        self.topology
    }
    pub fn backend(&self) -> CommBackend {
        self.backend
    }
    pub fn send_buffer_capacity(&self) -> usize {
        self.send_buffer_capacity
    }
    pub fn recv_buffer_capacity(&self) -> usize {
        self.recv_buffer_capacity
    }
    pub fn async_mode(&self) -> bool {
        self.async_mode
    }
}

impl Default for CommConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

impl fmt::Display for CommConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CommConfig(workers={}, topology={}, backend={}, async={})",
            self.num_workers, self.topology, self.backend, self.async_mode
        )
    }
}

// ── AllReduce ──────────────────────────────────────────────────────────

/// All-reduce collective: reduces data across all workers and distributes
/// the result so every worker holds the same reduced output.
#[derive(Debug, Clone)]
pub struct AllReduce {
    num_workers: usize,
    op: ReduceOp,
}

impl AllReduce {
    /// Create an all-reduce operation for the given worker count and op.
    pub fn new(num_workers: usize, op: ReduceOp) -> Self {
        Self { num_workers, op }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
    pub fn op(&self) -> ReduceOp {
        self.op
    }

    /// Execute the all-reduce on simulated per-worker data.
    ///
    /// Each inner `Vec<f32>` is one worker's contribution; all must have
    /// the same length.  Returns the reduced vector.
    pub fn execute(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>, CommError> {
        if inputs.len() != self.num_workers {
            return Err(CommError::SizeMismatch {
                expected: self.num_workers,
                actual: inputs.len(),
            });
        }
        let len = inputs[0].len();
        for (i, v) in inputs.iter().enumerate() {
            if v.len() != len {
                return Err(CommError::SizeMismatch { expected: len, actual: v.len() });
            }
            if i >= self.num_workers {
                return Err(CommError::InvalidWorker(i));
            }
        }

        let mut result = vec![0.0_f32; len];
        match self.op {
            ReduceOp::Sum => {
                for v in inputs {
                    for (r, x) in result.iter_mut().zip(v.iter()) {
                        *r += x;
                    }
                }
            }
            ReduceOp::Mean => {
                for v in inputs {
                    for (r, x) in result.iter_mut().zip(v.iter()) {
                        *r += x;
                    }
                }
                let n = self.num_workers as f32;
                for r in &mut result {
                    *r /= n;
                }
            }
            ReduceOp::Max => {
                result.copy_from_slice(&inputs[0]);
                for v in &inputs[1..] {
                    for (r, x) in result.iter_mut().zip(v.iter()) {
                        if *x > *r {
                            *r = *x;
                        }
                    }
                }
            }
            ReduceOp::Min => {
                result.copy_from_slice(&inputs[0]);
                for v in &inputs[1..] {
                    for (r, x) in result.iter_mut().zip(v.iter()) {
                        if *x < *r {
                            *r = *x;
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

impl fmt::Display for AllReduce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllReduce(workers={}, op={})", self.num_workers, self.op)
    }
}

// ── AllGather ──────────────────────────────────────────────────────────

/// All-gather collective: each worker contributes a chunk and every worker
/// receives the concatenation of all chunks.
#[derive(Debug, Clone)]
pub struct AllGather {
    num_workers: usize,
}

impl AllGather {
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Execute the all-gather on simulated per-worker chunks.
    ///
    /// Returns the concatenated result that every worker would receive.
    pub fn execute(&self, chunks: &[Vec<f32>]) -> Result<Vec<f32>, CommError> {
        if chunks.len() != self.num_workers {
            return Err(CommError::SizeMismatch {
                expected: self.num_workers,
                actual: chunks.len(),
            });
        }
        let mut gathered = Vec::new();
        for c in chunks {
            gathered.extend_from_slice(c);
        }
        Ok(gathered)
    }
}

impl fmt::Display for AllGather {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllGather(workers={})", self.num_workers)
    }
}

// ── ReduceScatter ──────────────────────────────────────────────────────

/// Reduce-scatter collective: reduces across all workers then scatters
/// equal-sized chunks so each worker gets a unique portion of the result.
#[derive(Debug, Clone)]
pub struct ReduceScatter {
    num_workers: usize,
    op: ReduceOp,
}

impl ReduceScatter {
    pub fn new(num_workers: usize, op: ReduceOp) -> Self {
        Self { num_workers, op }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
    pub fn op(&self) -> ReduceOp {
        self.op
    }

    /// Execute the reduce-scatter.
    ///
    /// Each worker provides data of the same length which must be evenly
    /// divisible by `num_workers`. Returns per-worker scattered chunks.
    pub fn execute(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, CommError> {
        if inputs.len() != self.num_workers {
            return Err(CommError::SizeMismatch {
                expected: self.num_workers,
                actual: inputs.len(),
            });
        }
        let len = inputs[0].len();
        if len % self.num_workers != 0 {
            return Err(CommError::SizeMismatch { expected: len, actual: len });
        }
        for v in inputs {
            if v.len() != len {
                return Err(CommError::SizeMismatch { expected: len, actual: v.len() });
            }
        }

        // Reduce (sum/mean/max/min).
        let reduce = AllReduce::new(self.num_workers, self.op);
        let reduced = reduce.execute(inputs)?;

        // Scatter: split into equal chunks.
        let chunk_size = len / self.num_workers;
        let scattered: Vec<Vec<f32>> = reduced.chunks(chunk_size).map(|c| c.to_vec()).collect();
        Ok(scattered)
    }
}

impl fmt::Display for ReduceScatter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReduceScatter(workers={}, op={})", self.num_workers, self.op)
    }
}

// ── Broadcast ──────────────────────────────────────────────────────────

/// Broadcast collective: root worker sends its data to all other workers.
#[derive(Debug, Clone)]
pub struct Broadcast {
    num_workers: usize,
    root: usize,
}

impl Broadcast {
    /// Create a broadcast from the given root worker.
    pub fn new(num_workers: usize, root: usize) -> Result<Self, CommError> {
        if root >= num_workers {
            return Err(CommError::InvalidWorker(root));
        }
        Ok(Self { num_workers, root })
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
    pub fn root(&self) -> usize {
        self.root
    }

    /// Execute the broadcast: returns `num_workers` copies of `root_data`.
    pub fn execute(&self, root_data: &[f32]) -> Vec<Vec<f32>> {
        (0..self.num_workers).map(|_| root_data.to_vec()).collect()
    }
}

impl fmt::Display for Broadcast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Broadcast(workers={}, root={})", self.num_workers, self.root)
    }
}

// ── RingTopology ───────────────────────────────────────────────────────

/// Ring communication topology where each worker sends to its right
/// neighbour and receives from its left neighbour.
#[derive(Debug, Clone)]
pub struct RingTopology {
    num_workers: usize,
}

impl RingTopology {
    /// Create a ring topology with the given number of workers.
    pub fn new(num_workers: usize) -> Result<Self, CommError> {
        if num_workers == 0 {
            return Err(CommError::EmptyTopology);
        }
        Ok(Self { num_workers })
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Return the right neighbour (send target) for the given worker.
    pub fn right_neighbour(&self, worker: usize) -> Result<usize, CommError> {
        if worker >= self.num_workers {
            return Err(CommError::InvalidWorker(worker));
        }
        Ok((worker + 1) % self.num_workers)
    }

    /// Return the left neighbour (receive source) for the given worker.
    pub fn left_neighbour(&self, worker: usize) -> Result<usize, CommError> {
        if worker >= self.num_workers {
            return Err(CommError::InvalidWorker(worker));
        }
        Ok((worker + self.num_workers - 1) % self.num_workers)
    }

    /// Return all (sender, receiver) edges in the ring.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        (0..self.num_workers).map(|w| (w, (w + 1) % self.num_workers)).collect()
    }

    /// Number of steps required for a full ring all-reduce.
    pub fn steps_for_allreduce(&self) -> usize {
        if self.num_workers <= 1 {
            return 0;
        }
        // Ring all-reduce: 2*(N-1) steps (reduce-scatter + all-gather).
        2 * (self.num_workers - 1)
    }
}

impl fmt::Display for RingTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingTopology(workers={})", self.num_workers)
    }
}

// ── TreeTopology ───────────────────────────────────────────────────────

/// Tree (binary) communication topology for hierarchical reduce/broadcast.
///
/// Worker 0 is the root. Each worker `i` has children `2*i + 1` and
/// `2*i + 2` (if they exist).
#[derive(Debug, Clone)]
pub struct TreeTopology {
    num_workers: usize,
}

impl TreeTopology {
    /// Create a binary-tree topology with the given number of workers.
    pub fn new(num_workers: usize) -> Result<Self, CommError> {
        if num_workers == 0 {
            return Err(CommError::EmptyTopology);
        }
        Ok(Self { num_workers })
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Return the parent of the given worker, or `None` for root (0).
    pub fn parent(&self, worker: usize) -> Result<Option<usize>, CommError> {
        if worker >= self.num_workers {
            return Err(CommError::InvalidWorker(worker));
        }
        if worker == 0 { Ok(None) } else { Ok(Some((worker - 1) / 2)) }
    }

    /// Return the children of the given worker (0, 1, or 2 children).
    pub fn children(&self, worker: usize) -> Result<Vec<usize>, CommError> {
        if worker >= self.num_workers {
            return Err(CommError::InvalidWorker(worker));
        }
        let mut kids = Vec::new();
        let left = 2 * worker + 1;
        let right = 2 * worker + 2;
        if left < self.num_workers {
            kids.push(left);
        }
        if right < self.num_workers {
            kids.push(right);
        }
        Ok(kids)
    }

    /// Depth of the tree (0-indexed).
    pub fn depth(&self) -> usize {
        if self.num_workers <= 1 {
            return 0;
        }
        (self.num_workers as f64).log2().ceil() as usize
    }

    /// Whether the given worker is a leaf node.
    pub fn is_leaf(&self, worker: usize) -> Result<bool, CommError> {
        Ok(self.children(worker)?.is_empty())
    }

    /// Return all (parent, child) edges in the tree.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();
        for w in 0..self.num_workers {
            let left = 2 * w + 1;
            let right = 2 * w + 2;
            if left < self.num_workers {
                edges.push((w, left));
            }
            if right < self.num_workers {
                edges.push((w, right));
            }
        }
        edges
    }
}

impl fmt::Display for TreeTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TreeTopology(workers={})", self.num_workers)
    }
}

// ── CommBuffer ─────────────────────────────────────────────────────────

/// Double-buffered communication buffer for overlapping communication
/// with computation.
///
/// Two internal buffers alternate roles: while one is being filled
/// (written to), the other is available for reading (draining).
#[derive(Debug, Clone)]
pub struct CommBuffer {
    capacity: usize,
    front: Vec<f32>,
    back: Vec<f32>,
    /// `true` when `front` is the write target, `back` is readable.
    front_is_write: bool,
}

impl CommBuffer {
    /// Create a double buffer with the given per-buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            front: Vec::with_capacity(capacity),
            back: Vec::with_capacity(capacity),
            front_is_write: true,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Write data into the current write buffer, returning how many
    /// elements were actually written (capped at remaining capacity).
    pub fn write(&mut self, data: &[f32]) -> usize {
        let cap = self.capacity;
        let buf = self.write_buf_mut();
        let remaining = cap.saturating_sub(buf.len());
        let n = data.len().min(remaining);
        buf.extend_from_slice(&data[..n]);
        n
    }

    /// Read (drain) the current read buffer, returning its contents.
    pub fn read(&mut self) -> Vec<f32> {
        let buf = self.read_buf_mut();
        std::mem::take(buf)
    }

    /// Swap the roles of the two buffers.
    pub fn swap(&mut self) {
        self.front_is_write = !self.front_is_write;
    }

    /// Number of elements in the write buffer.
    pub fn write_len(&self) -> usize {
        if self.front_is_write { self.front.len() } else { self.back.len() }
    }

    /// Number of elements in the read buffer.
    pub fn read_len(&self) -> usize {
        if self.front_is_write { self.back.len() } else { self.front.len() }
    }

    /// Clear both buffers.
    pub fn clear(&mut self) {
        self.front.clear();
        self.back.clear();
    }

    fn write_buf_mut(&mut self) -> &mut Vec<f32> {
        if self.front_is_write { &mut self.front } else { &mut self.back }
    }

    fn read_buf_mut(&mut self) -> &mut Vec<f32> {
        if self.front_is_write { &mut self.back } else { &mut self.front }
    }
}

impl fmt::Display for CommBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CommBuffer(cap={}, write_len={}, read_len={})",
            self.capacity,
            self.write_len(),
            self.read_len()
        )
    }
}

// ── CommProfiler ───────────────────────────────────────────────────────

/// Records for a single profiled communication event.
#[derive(Debug, Clone)]
pub struct CommEvent {
    /// Label identifying the operation (e.g. "allreduce").
    pub label: String,
    /// Duration of the operation.
    pub duration: Duration,
    /// Number of elements transferred.
    pub elements: usize,
}

/// Profiles communication latency, bandwidth, and overlap efficiency.
#[derive(Debug, Clone)]
pub struct CommProfiler {
    events: Vec<CommEvent>,
    compute_time: Duration,
    comm_time: Duration,
}

impl CommProfiler {
    pub fn new() -> Self {
        Self { events: Vec::new(), compute_time: Duration::ZERO, comm_time: Duration::ZERO }
    }

    /// Record a communication event.
    pub fn record(&mut self, label: impl Into<String>, duration: Duration, elements: usize) {
        let label = label.into();
        self.comm_time += duration;
        self.events.push(CommEvent { label, duration, elements });
    }

    /// Record time spent on computation (non-communication).
    pub fn record_compute(&mut self, duration: Duration) {
        self.compute_time += duration;
    }

    /// Total communication time recorded.
    pub fn total_comm_time(&self) -> Duration {
        self.comm_time
    }

    /// Total compute time recorded.
    pub fn total_compute_time(&self) -> Duration {
        self.compute_time
    }

    /// Number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// All recorded events.
    pub fn events(&self) -> &[CommEvent] {
        &self.events
    }

    /// Total elements transferred across all events.
    pub fn total_elements(&self) -> usize {
        self.events.iter().map(|e| e.elements).sum()
    }

    /// Effective bandwidth in elements per second (returns 0.0 if no time).
    pub fn bandwidth_elements_per_sec(&self) -> f64 {
        let secs = self.comm_time.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        self.total_elements() as f64 / secs
    }

    /// Communication-computation overlap efficiency in `[0.0, 1.0]`.
    ///
    /// Returns the fraction of wall-clock time that is overlapped:
    /// `overlap = 1.0 - wall / (compute + comm)` when both are nonzero.
    /// A value of 0.0 means fully serialised; 1.0 is theoretical maximum.
    pub fn overlap_efficiency(&self) -> f64 {
        let total = self.compute_time + self.comm_time;
        if total.is_zero() {
            return 0.0;
        }
        let wall = self.compute_time.max(self.comm_time);
        1.0 - (wall.as_secs_f64() / total.as_secs_f64())
    }

    /// Average latency per event.
    pub fn avg_latency(&self) -> Duration {
        if self.events.is_empty() {
            return Duration::ZERO;
        }
        self.comm_time / self.events.len() as u32
    }

    /// Clear all recorded data.
    pub fn reset(&mut self) {
        self.events.clear();
        self.compute_time = Duration::ZERO;
        self.comm_time = Duration::ZERO;
    }

    /// Per-label aggregate: (total_duration, total_elements, count).
    pub fn per_label_summary(&self) -> HashMap<String, (Duration, usize, usize)> {
        let mut map: HashMap<String, (Duration, usize, usize)> = HashMap::new();
        for e in &self.events {
            let entry = map.entry(e.label.clone()).or_insert((Duration::ZERO, 0, 0));
            entry.0 += e.duration;
            entry.1 += e.elements;
            entry.2 += 1;
        }
        map
    }
}

impl Default for CommProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CommProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CommProfiler(events={}, comm={:?}, compute={:?}, overlap={:.2})",
            self.events.len(),
            self.comm_time,
            self.compute_time,
            self.overlap_efficiency()
        )
    }
}

// ── ParallelCommEngine ─────────────────────────────────────────────────

/// Unified communication engine for distributed inference.
///
/// Wraps [`CommConfig`], topology, buffers, and profiler into a single
/// entry point for all collective operations.
#[derive(Debug, Clone)]
pub struct ParallelCommEngine {
    config: CommConfig,
    ring: Option<RingTopology>,
    tree: Option<TreeTopology>,
    buffers: Vec<CommBuffer>,
    profiler: CommProfiler,
}

impl ParallelCommEngine {
    /// Create an engine from the given configuration.
    pub fn new(config: CommConfig) -> Result<Self, CommError> {
        if config.num_workers() == 0 {
            return Err(CommError::EmptyTopology);
        }
        let ring = match config.topology() {
            TopologyKind::Ring => Some(RingTopology::new(config.num_workers())?),
            TopologyKind::Tree => None,
        };
        let tree = match config.topology() {
            TopologyKind::Tree => Some(TreeTopology::new(config.num_workers())?),
            TopologyKind::Ring => None,
        };
        let buffers = (0..config.num_workers())
            .map(|_| CommBuffer::new(config.send_buffer_capacity()))
            .collect();
        Ok(Self { config, ring, tree, buffers, profiler: CommProfiler::new() })
    }

    pub fn config(&self) -> &CommConfig {
        &self.config
    }
    pub fn profiler(&self) -> &CommProfiler {
        &self.profiler
    }
    pub fn profiler_mut(&mut self) -> &mut CommProfiler {
        &mut self.profiler
    }
    pub fn ring_topology(&self) -> Option<&RingTopology> {
        self.ring.as_ref()
    }
    pub fn tree_topology(&self) -> Option<&TreeTopology> {
        self.tree.as_ref()
    }

    /// Access the communication buffer for a given worker.
    pub fn buffer(&self, worker: usize) -> Result<&CommBuffer, CommError> {
        self.buffers.get(worker).ok_or(CommError::InvalidWorker(worker))
    }

    /// Access the communication buffer mutably.
    pub fn buffer_mut(&mut self, worker: usize) -> Result<&mut CommBuffer, CommError> {
        if worker >= self.buffers.len() {
            return Err(CommError::InvalidWorker(worker));
        }
        Ok(&mut self.buffers[worker])
    }

    /// Perform an all-reduce across simulated worker inputs.
    pub fn all_reduce(&mut self, inputs: &[Vec<f32>], op: ReduceOp) -> Result<Vec<f32>, CommError> {
        let start = Instant::now();
        let ar = AllReduce::new(self.config.num_workers(), op);
        let result = ar.execute(inputs)?;
        let elapsed = start.elapsed();
        let elems: usize = inputs.iter().map(|v| v.len()).sum();
        self.profiler.record("allreduce", elapsed, elems);
        Ok(result)
    }

    /// Perform an all-gather across simulated worker chunks.
    pub fn all_gather(&mut self, chunks: &[Vec<f32>]) -> Result<Vec<f32>, CommError> {
        let start = Instant::now();
        let ag = AllGather::new(self.config.num_workers());
        let result = ag.execute(chunks)?;
        let elapsed = start.elapsed();
        let elems: usize = chunks.iter().map(|v| v.len()).sum();
        self.profiler.record("allgather", elapsed, elems);
        Ok(result)
    }

    /// Perform a reduce-scatter across simulated worker inputs.
    pub fn reduce_scatter(
        &mut self,
        inputs: &[Vec<f32>],
        op: ReduceOp,
    ) -> Result<Vec<Vec<f32>>, CommError> {
        let start = Instant::now();
        let rs = ReduceScatter::new(self.config.num_workers(), op);
        let result = rs.execute(inputs)?;
        let elapsed = start.elapsed();
        let elems: usize = inputs.iter().map(|v| v.len()).sum();
        self.profiler.record("reduce_scatter", elapsed, elems);
        Ok(result)
    }

    /// Broadcast data from the given root worker.
    pub fn broadcast(&mut self, root: usize, data: &[f32]) -> Result<Vec<Vec<f32>>, CommError> {
        let start = Instant::now();
        let bc = Broadcast::new(self.config.num_workers(), root)?;
        let result = bc.execute(data);
        let elapsed = start.elapsed();
        self.profiler.record("broadcast", elapsed, data.len());
        Ok(result)
    }

    /// Reset the profiler.
    pub fn reset_profiler(&mut self) {
        self.profiler.reset();
    }

    /// Number of workers in the engine.
    pub fn num_workers(&self) -> usize {
        self.config.num_workers()
    }
}

impl fmt::Display for ParallelCommEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParallelCommEngine({})", self.config)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CommConfig tests ───────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let c = CommConfig::default();
        assert_eq!(c.num_workers(), 1);
        assert_eq!(c.topology(), TopologyKind::Ring);
        assert_eq!(c.backend(), CommBackend::Simulated);
        assert!(!c.async_mode());
    }

    #[test]
    fn test_config_new() {
        let c = CommConfig::new(8);
        assert_eq!(c.num_workers(), 8);
        assert_eq!(c.send_buffer_capacity(), 4096);
        assert_eq!(c.recv_buffer_capacity(), 4096);
    }

    #[test]
    fn test_config_builder() {
        let c = CommConfig::new(4)
            .with_topology(TopologyKind::Tree)
            .with_backend(CommBackend::SharedMemory)
            .with_send_buffer_capacity(1024)
            .with_recv_buffer_capacity(2048)
            .with_async_mode(true);
        assert_eq!(c.topology(), TopologyKind::Tree);
        assert_eq!(c.backend(), CommBackend::SharedMemory);
        assert_eq!(c.send_buffer_capacity(), 1024);
        assert_eq!(c.recv_buffer_capacity(), 2048);
        assert!(c.async_mode());
    }

    #[test]
    fn test_config_display() {
        let c = CommConfig::new(2);
        let s = format!("{c}");
        assert!(s.contains("workers=2"));
        assert!(s.contains("Ring"));
    }

    #[test]
    fn test_config_clone() {
        let c = CommConfig::new(4).with_async_mode(true);
        let c2 = c.clone();
        assert_eq!(c2.num_workers(), 4);
        assert!(c2.async_mode());
    }

    #[test]
    fn test_config_debug() {
        let c = CommConfig::new(1);
        let dbg = format!("{c:?}");
        assert!(dbg.contains("CommConfig"));
    }

    // ── ReduceOp / CommBackend / TopologyKind display ──────────────

    #[test]
    fn test_reduce_op_display() {
        assert_eq!(format!("{}", ReduceOp::Sum), "Sum");
        assert_eq!(format!("{}", ReduceOp::Mean), "Mean");
        assert_eq!(format!("{}", ReduceOp::Max), "Max");
        assert_eq!(format!("{}", ReduceOp::Min), "Min");
    }

    #[test]
    fn test_comm_backend_display() {
        assert_eq!(format!("{}", CommBackend::Simulated), "Simulated");
        assert_eq!(format!("{}", CommBackend::SharedMemory), "SharedMemory");
        assert_eq!(format!("{}", CommBackend::Network), "Network");
    }

    #[test]
    fn test_comm_backend_default() {
        assert_eq!(CommBackend::default(), CommBackend::Simulated);
    }

    #[test]
    fn test_topology_kind_display() {
        assert_eq!(format!("{}", TopologyKind::Ring), "Ring");
        assert_eq!(format!("{}", TopologyKind::Tree), "Tree");
    }

    #[test]
    fn test_topology_kind_default() {
        assert_eq!(TopologyKind::default(), TopologyKind::Ring);
    }

    // ── CommError display ──────────────────────────────────────────

    #[test]
    fn test_error_invalid_worker() {
        let e = CommError::InvalidWorker(5);
        assert_eq!(format!("{e}"), "invalid worker id: 5");
    }

    #[test]
    fn test_error_size_mismatch() {
        let e = CommError::SizeMismatch { expected: 4, actual: 3 };
        assert!(format!("{e}").contains("expected 4"));
    }

    #[test]
    fn test_error_buffer_not_ready() {
        let e = CommError::BufferNotReady;
        assert!(format!("{e}").contains("not ready"));
    }

    #[test]
    fn test_error_empty_topology() {
        let e = CommError::EmptyTopology;
        assert!(format!("{e}").contains("no workers"));
    }

    #[test]
    fn test_error_unsupported() {
        let e = CommError::UnsupportedOperation("foo".into());
        assert!(format!("{e}").contains("foo"));
    }

    // ── AllReduce tests ────────────────────────────────────────────

    #[test]
    fn test_allreduce_sum_two_workers() {
        let ar = AllReduce::new(2, ReduceOp::Sum);
        let r = ar.execute(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(r, vec![4.0, 6.0]);
    }

    #[test]
    fn test_allreduce_sum_single_worker() {
        let ar = AllReduce::new(1, ReduceOp::Sum);
        let r = ar.execute(&[vec![5.0, 10.0]]).unwrap();
        assert_eq!(r, vec![5.0, 10.0]);
    }

    #[test]
    fn test_allreduce_mean() {
        let ar = AllReduce::new(2, ReduceOp::Mean);
        let r = ar.execute(&[vec![2.0, 8.0], vec![4.0, 6.0]]).unwrap();
        assert_eq!(r, vec![3.0, 7.0]);
    }

    #[test]
    fn test_allreduce_max() {
        let ar = AllReduce::new(3, ReduceOp::Max);
        let r = ar.execute(&[vec![1.0, 9.0], vec![5.0, 2.0], vec![3.0, 7.0]]).unwrap();
        assert_eq!(r, vec![5.0, 9.0]);
    }

    #[test]
    fn test_allreduce_min() {
        let ar = AllReduce::new(3, ReduceOp::Min);
        let r = ar.execute(&[vec![1.0, 9.0], vec![5.0, 2.0], vec![3.0, 7.0]]).unwrap();
        assert_eq!(r, vec![1.0, 2.0]);
    }

    #[test]
    fn test_allreduce_wrong_worker_count() {
        let ar = AllReduce::new(2, ReduceOp::Sum);
        let r = ar.execute(&[vec![1.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn test_allreduce_mismatched_lengths() {
        let ar = AllReduce::new(2, ReduceOp::Sum);
        let r = ar.execute(&[vec![1.0, 2.0], vec![3.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn test_allreduce_display() {
        let ar = AllReduce::new(4, ReduceOp::Sum);
        assert!(format!("{ar}").contains("workers=4"));
    }

    #[test]
    fn test_allreduce_accessors() {
        let ar = AllReduce::new(3, ReduceOp::Max);
        assert_eq!(ar.num_workers(), 3);
        assert_eq!(ar.op(), ReduceOp::Max);
    }

    #[test]
    fn test_allreduce_sum_four_workers() {
        let ar = AllReduce::new(4, ReduceOp::Sum);
        let r = ar.execute(&[vec![1.0], vec![2.0], vec![3.0], vec![4.0]]).unwrap();
        assert_eq!(r, vec![10.0]);
    }

    #[test]
    fn test_allreduce_mean_four_workers() {
        let ar = AllReduce::new(4, ReduceOp::Mean);
        let r = ar.execute(&[vec![4.0], vec![8.0], vec![12.0], vec![16.0]]).unwrap();
        assert_eq!(r, vec![10.0]);
    }

    #[test]
    fn test_allreduce_empty_data() {
        let ar = AllReduce::new(2, ReduceOp::Sum);
        let r = ar.execute(&[vec![], vec![]]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn test_allreduce_clone() {
        let ar = AllReduce::new(2, ReduceOp::Sum);
        let ar2 = ar.clone();
        assert_eq!(ar2.num_workers(), 2);
    }

    // ── AllGather tests ────────────────────────────────────────────

    #[test]
    fn test_allgather_basic() {
        let ag = AllGather::new(3);
        let r = ag.execute(&[vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_allgather_multi_element_chunks() {
        let ag = AllGather::new(2);
        let r = ag.execute(&[vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(r, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_allgather_single_worker() {
        let ag = AllGather::new(1);
        let r = ag.execute(&[vec![42.0]]).unwrap();
        assert_eq!(r, vec![42.0]);
    }

    #[test]
    fn test_allgather_wrong_count() {
        let ag = AllGather::new(2);
        let r = ag.execute(&[vec![1.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn test_allgather_display() {
        let ag = AllGather::new(5);
        assert!(format!("{ag}").contains("workers=5"));
    }

    #[test]
    fn test_allgather_accessors() {
        let ag = AllGather::new(7);
        assert_eq!(ag.num_workers(), 7);
    }

    #[test]
    fn test_allgather_empty_chunks() {
        let ag = AllGather::new(2);
        let r = ag.execute(&[vec![], vec![]]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn test_allgather_clone() {
        let ag = AllGather::new(3);
        let ag2 = ag.clone();
        assert_eq!(ag2.num_workers(), 3);
    }

    // ── ReduceScatter tests ────────────────────────────────────────

    #[test]
    fn test_reduce_scatter_sum() {
        let rs = ReduceScatter::new(2, ReduceOp::Sum);
        let r = rs.execute(&[vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0], vec![6.0, 8.0]);
        assert_eq!(r[1], vec![10.0, 12.0]);
    }

    #[test]
    fn test_reduce_scatter_mean() {
        let rs = ReduceScatter::new(2, ReduceOp::Mean);
        let r = rs.execute(&[vec![2.0, 4.0], vec![6.0, 8.0]]).unwrap();
        assert_eq!(r[0], vec![4.0]);
        assert_eq!(r[1], vec![6.0]);
    }

    #[test]
    fn test_reduce_scatter_wrong_count() {
        let rs = ReduceScatter::new(3, ReduceOp::Sum);
        let r = rs.execute(&[vec![1.0, 2.0, 3.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn test_reduce_scatter_not_divisible() {
        let rs = ReduceScatter::new(2, ReduceOp::Sum);
        // length 3 not divisible by 2
        let r = rs.execute(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn test_reduce_scatter_display() {
        let rs = ReduceScatter::new(4, ReduceOp::Max);
        let s = format!("{rs}");
        assert!(s.contains("workers=4"));
        assert!(s.contains("Max"));
    }

    #[test]
    fn test_reduce_scatter_accessors() {
        let rs = ReduceScatter::new(2, ReduceOp::Min);
        assert_eq!(rs.num_workers(), 2);
        assert_eq!(rs.op(), ReduceOp::Min);
    }

    #[test]
    fn test_reduce_scatter_mismatched_lens() {
        let rs = ReduceScatter::new(2, ReduceOp::Sum);
        let r = rs.execute(&[vec![1.0, 2.0], vec![3.0]]);
        assert!(r.is_err());
    }

    #[test]
    fn test_reduce_scatter_clone() {
        let rs = ReduceScatter::new(2, ReduceOp::Sum);
        let rs2 = rs.clone();
        assert_eq!(rs2.num_workers(), 2);
    }

    // ── Broadcast tests ────────────────────────────────────────────

    #[test]
    fn test_broadcast_basic() {
        let bc = Broadcast::new(3, 0).unwrap();
        let r = bc.execute(&[10.0, 20.0]);
        assert_eq!(r.len(), 3);
        for v in &r {
            assert_eq!(v, &vec![10.0, 20.0]);
        }
    }

    #[test]
    fn test_broadcast_non_root() {
        let bc = Broadcast::new(4, 2).unwrap();
        assert_eq!(bc.root(), 2);
        let r = bc.execute(&[7.0]);
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_broadcast_invalid_root() {
        let r = Broadcast::new(3, 5);
        assert!(r.is_err());
    }

    #[test]
    fn test_broadcast_single_worker() {
        let bc = Broadcast::new(1, 0).unwrap();
        let r = bc.execute(&[1.0, 2.0, 3.0]);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_display() {
        let bc = Broadcast::new(2, 0).unwrap();
        let s = format!("{bc}");
        assert!(s.contains("root=0"));
    }

    #[test]
    fn test_broadcast_accessors() {
        let bc = Broadcast::new(3, 1).unwrap();
        assert_eq!(bc.num_workers(), 3);
        assert_eq!(bc.root(), 1);
    }

    #[test]
    fn test_broadcast_empty_data() {
        let bc = Broadcast::new(2, 0).unwrap();
        let r = bc.execute(&[]);
        assert_eq!(r.len(), 2);
        assert!(r[0].is_empty());
    }

    #[test]
    fn test_broadcast_clone() {
        let bc = Broadcast::new(3, 0).unwrap();
        let bc2 = bc.clone();
        assert_eq!(bc2.num_workers(), 3);
    }

    // ── RingTopology tests ─────────────────────────────────────────

    #[test]
    fn test_ring_new() {
        let r = RingTopology::new(4).unwrap();
        assert_eq!(r.num_workers(), 4);
    }

    #[test]
    fn test_ring_empty() {
        assert!(RingTopology::new(0).is_err());
    }

    #[test]
    fn test_ring_right_neighbour() {
        let r = RingTopology::new(4).unwrap();
        assert_eq!(r.right_neighbour(0).unwrap(), 1);
        assert_eq!(r.right_neighbour(3).unwrap(), 0);
    }

    #[test]
    fn test_ring_left_neighbour() {
        let r = RingTopology::new(4).unwrap();
        assert_eq!(r.left_neighbour(0).unwrap(), 3);
        assert_eq!(r.left_neighbour(1).unwrap(), 0);
    }

    #[test]
    fn test_ring_invalid_worker() {
        let r = RingTopology::new(3).unwrap();
        assert!(r.right_neighbour(5).is_err());
        assert!(r.left_neighbour(5).is_err());
    }

    #[test]
    fn test_ring_edges() {
        let r = RingTopology::new(3).unwrap();
        let e = r.edges();
        assert_eq!(e, vec![(0, 1), (1, 2), (2, 0)]);
    }

    #[test]
    fn test_ring_single_worker_edges() {
        let r = RingTopology::new(1).unwrap();
        assert_eq!(r.edges(), vec![(0, 0)]);
    }

    #[test]
    fn test_ring_steps_allreduce() {
        let r = RingTopology::new(4).unwrap();
        assert_eq!(r.steps_for_allreduce(), 6);
    }

    #[test]
    fn test_ring_steps_single() {
        let r = RingTopology::new(1).unwrap();
        assert_eq!(r.steps_for_allreduce(), 0);
    }

    #[test]
    fn test_ring_display() {
        let r = RingTopology::new(3).unwrap();
        assert!(format!("{r}").contains("workers=3"));
    }

    #[test]
    fn test_ring_clone() {
        let r = RingTopology::new(4).unwrap();
        let r2 = r.clone();
        assert_eq!(r2.num_workers(), 4);
    }

    #[test]
    fn test_ring_two_workers() {
        let r = RingTopology::new(2).unwrap();
        assert_eq!(r.right_neighbour(0).unwrap(), 1);
        assert_eq!(r.right_neighbour(1).unwrap(), 0);
        assert_eq!(r.left_neighbour(0).unwrap(), 1);
        assert_eq!(r.left_neighbour(1).unwrap(), 0);
    }

    // ── TreeTopology tests ─────────────────────────────────────────

    #[test]
    fn test_tree_new() {
        let t = TreeTopology::new(7).unwrap();
        assert_eq!(t.num_workers(), 7);
    }

    #[test]
    fn test_tree_empty() {
        assert!(TreeTopology::new(0).is_err());
    }

    #[test]
    fn test_tree_parent_root() {
        let t = TreeTopology::new(7).unwrap();
        assert_eq!(t.parent(0).unwrap(), None);
    }

    #[test]
    fn test_tree_parent_child() {
        let t = TreeTopology::new(7).unwrap();
        assert_eq!(t.parent(1).unwrap(), Some(0));
        assert_eq!(t.parent(2).unwrap(), Some(0));
        assert_eq!(t.parent(3).unwrap(), Some(1));
        assert_eq!(t.parent(6).unwrap(), Some(2));
    }

    #[test]
    fn test_tree_children() {
        let t = TreeTopology::new(7).unwrap();
        assert_eq!(t.children(0).unwrap(), vec![1, 2]);
        assert_eq!(t.children(1).unwrap(), vec![3, 4]);
        assert_eq!(t.children(3).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn test_tree_children_partial() {
        let t = TreeTopology::new(4).unwrap();
        // worker 1 has only left child (3), right child (4) out of range
        assert_eq!(t.children(1).unwrap(), vec![3]);
    }

    #[test]
    fn test_tree_is_leaf() {
        let t = TreeTopology::new(7).unwrap();
        assert!(!t.is_leaf(0).unwrap());
        assert!(t.is_leaf(3).unwrap());
        assert!(t.is_leaf(6).unwrap());
    }

    #[test]
    fn test_tree_depth() {
        let t1 = TreeTopology::new(1).unwrap();
        assert_eq!(t1.depth(), 0);
        let t7 = TreeTopology::new(7).unwrap();
        assert_eq!(t7.depth(), 3);
    }

    #[test]
    fn test_tree_edges() {
        let t = TreeTopology::new(3).unwrap();
        let e = t.edges();
        assert_eq!(e, vec![(0, 1), (0, 2)]);
    }

    #[test]
    fn test_tree_invalid_worker() {
        let t = TreeTopology::new(3).unwrap();
        assert!(t.parent(10).is_err());
        assert!(t.children(10).is_err());
    }

    #[test]
    fn test_tree_display() {
        let t = TreeTopology::new(5).unwrap();
        assert!(format!("{t}").contains("workers=5"));
    }

    #[test]
    fn test_tree_clone() {
        let t = TreeTopology::new(4).unwrap();
        let t2 = t.clone();
        assert_eq!(t2.num_workers(), 4);
    }

    #[test]
    fn test_tree_single_worker() {
        let t = TreeTopology::new(1).unwrap();
        assert!(t.is_leaf(0).unwrap());
        assert_eq!(t.children(0).unwrap(), Vec::<usize>::new());
        assert_eq!(t.edges(), Vec::<(usize, usize)>::new());
    }

    // ── CommBuffer tests ───────────────────────────────────────────

    #[test]
    fn test_buffer_new() {
        let b = CommBuffer::new(16);
        assert_eq!(b.capacity(), 16);
        assert_eq!(b.write_len(), 0);
        assert_eq!(b.read_len(), 0);
    }

    #[test]
    fn test_buffer_write_read() {
        let mut b = CommBuffer::new(8);
        let n = b.write(&[1.0, 2.0, 3.0]);
        assert_eq!(n, 3);
        assert_eq!(b.write_len(), 3);
        // swap so data moves to read side
        b.swap();
        let data = b.read();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_buffer_write_capped() {
        let mut b = CommBuffer::new(2);
        let n = b.write(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(n, 2);
        assert_eq!(b.write_len(), 2);
    }

    #[test]
    fn test_buffer_double_buffer_swap() {
        let mut b = CommBuffer::new(8);
        b.write(&[1.0, 2.0]);
        b.swap();
        b.write(&[10.0, 20.0]);
        // read should give the first batch
        let r = b.read();
        assert_eq!(r, vec![1.0, 2.0]);
        b.swap();
        let r2 = b.read();
        assert_eq!(r2, vec![10.0, 20.0]);
    }

    #[test]
    fn test_buffer_clear() {
        let mut b = CommBuffer::new(8);
        b.write(&[1.0]);
        b.swap();
        b.write(&[2.0]);
        b.clear();
        assert_eq!(b.write_len(), 0);
        assert_eq!(b.read_len(), 0);
    }

    #[test]
    fn test_buffer_display() {
        let b = CommBuffer::new(4);
        let s = format!("{b}");
        assert!(s.contains("cap=4"));
    }

    #[test]
    fn test_buffer_clone() {
        let mut b = CommBuffer::new(4);
        b.write(&[1.0]);
        let b2 = b.clone();
        assert_eq!(b2.write_len(), 1);
    }

    #[test]
    fn test_buffer_read_empty() {
        let mut b = CommBuffer::new(4);
        let data = b.read();
        assert!(data.is_empty());
    }

    #[test]
    fn test_buffer_multiple_writes() {
        let mut b = CommBuffer::new(8);
        b.write(&[1.0, 2.0]);
        b.write(&[3.0, 4.0]);
        assert_eq!(b.write_len(), 4);
    }

    // ── CommProfiler tests ─────────────────────────────────────────

    #[test]
    fn test_profiler_new() {
        let p = CommProfiler::new();
        assert_eq!(p.event_count(), 0);
        assert_eq!(p.total_comm_time(), Duration::ZERO);
    }

    #[test]
    fn test_profiler_default() {
        let p = CommProfiler::default();
        assert_eq!(p.event_count(), 0);
    }

    #[test]
    fn test_profiler_record() {
        let mut p = CommProfiler::new();
        p.record("allreduce", Duration::from_millis(10), 1024);
        assert_eq!(p.event_count(), 1);
        assert_eq!(p.total_elements(), 1024);
    }

    #[test]
    fn test_profiler_record_compute() {
        let mut p = CommProfiler::new();
        p.record_compute(Duration::from_millis(50));
        assert_eq!(p.total_compute_time(), Duration::from_millis(50));
    }

    #[test]
    fn test_profiler_bandwidth() {
        let mut p = CommProfiler::new();
        p.record("op", Duration::from_secs(1), 1000);
        let bw = p.bandwidth_elements_per_sec();
        assert!((bw - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_profiler_bandwidth_zero() {
        let p = CommProfiler::new();
        assert_eq!(p.bandwidth_elements_per_sec(), 0.0);
    }

    #[test]
    fn test_profiler_overlap_efficiency_zero() {
        let p = CommProfiler::new();
        assert_eq!(p.overlap_efficiency(), 0.0);
    }

    #[test]
    fn test_profiler_overlap_efficiency_serial() {
        let mut p = CommProfiler::new();
        p.record("op", Duration::from_millis(100), 10);
        p.record_compute(Duration::from_millis(100));
        // wall = max(100, 100) = 100; total = 200 ⇒ overlap = 1 - 100/200 = 0.5
        let eff = p.overlap_efficiency();
        assert!((eff - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_profiler_avg_latency() {
        let mut p = CommProfiler::new();
        p.record("a", Duration::from_millis(10), 100);
        p.record("b", Duration::from_millis(30), 200);
        let avg = p.avg_latency();
        assert_eq!(avg, Duration::from_millis(20));
    }

    #[test]
    fn test_profiler_avg_latency_empty() {
        let p = CommProfiler::new();
        assert_eq!(p.avg_latency(), Duration::ZERO);
    }

    #[test]
    fn test_profiler_reset() {
        let mut p = CommProfiler::new();
        p.record("x", Duration::from_millis(5), 50);
        p.record_compute(Duration::from_millis(10));
        p.reset();
        assert_eq!(p.event_count(), 0);
        assert_eq!(p.total_comm_time(), Duration::ZERO);
        assert_eq!(p.total_compute_time(), Duration::ZERO);
    }

    #[test]
    fn test_profiler_per_label_summary() {
        let mut p = CommProfiler::new();
        p.record("allreduce", Duration::from_millis(10), 100);
        p.record("allreduce", Duration::from_millis(20), 200);
        p.record("broadcast", Duration::from_millis(5), 50);
        let summary = p.per_label_summary();
        assert_eq!(summary.len(), 2);
        let (dur, elems, count) = &summary["allreduce"];
        assert_eq!(*dur, Duration::from_millis(30));
        assert_eq!(*elems, 300);
        assert_eq!(*count, 2);
    }

    #[test]
    fn test_profiler_events_accessor() {
        let mut p = CommProfiler::new();
        p.record("test", Duration::from_millis(1), 10);
        assert_eq!(p.events().len(), 1);
        assert_eq!(p.events()[0].label, "test");
    }

    #[test]
    fn test_profiler_display() {
        let p = CommProfiler::new();
        let s = format!("{p}");
        assert!(s.contains("CommProfiler"));
    }

    #[test]
    fn test_profiler_clone() {
        let mut p = CommProfiler::new();
        p.record("a", Duration::from_millis(1), 1);
        let p2 = p.clone();
        assert_eq!(p2.event_count(), 1);
    }

    // ── ParallelCommEngine tests ───────────────────────────────────

    #[test]
    fn test_engine_new_ring() {
        let cfg = CommConfig::new(4);
        let e = ParallelCommEngine::new(cfg).unwrap();
        assert_eq!(e.num_workers(), 4);
        assert!(e.ring_topology().is_some());
        assert!(e.tree_topology().is_none());
    }

    #[test]
    fn test_engine_new_tree() {
        let cfg = CommConfig::new(4).with_topology(TopologyKind::Tree);
        let e = ParallelCommEngine::new(cfg).unwrap();
        assert!(e.ring_topology().is_none());
        assert!(e.tree_topology().is_some());
    }

    #[test]
    fn test_engine_zero_workers() {
        let cfg = CommConfig::new(0);
        assert!(ParallelCommEngine::new(cfg).is_err());
    }

    #[test]
    fn test_engine_allreduce() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let r = e.all_reduce(&[vec![1.0, 2.0], vec![3.0, 4.0]], ReduceOp::Sum).unwrap();
        assert_eq!(r, vec![4.0, 6.0]);
        assert_eq!(e.profiler().event_count(), 1);
    }

    #[test]
    fn test_engine_allgather() {
        let cfg = CommConfig::new(3);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let r = e.all_gather(&[vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        assert_eq!(r, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_engine_reduce_scatter() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let r = e.reduce_scatter(&[vec![1.0, 2.0], vec![3.0, 4.0]], ReduceOp::Sum).unwrap();
        assert_eq!(r, vec![vec![4.0], vec![6.0]]);
    }

    #[test]
    fn test_engine_broadcast() {
        let cfg = CommConfig::new(3);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let r = e.broadcast(0, &[5.0, 6.0]).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0], vec![5.0, 6.0]);
    }

    #[test]
    fn test_engine_broadcast_invalid_root() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        assert!(e.broadcast(5, &[1.0]).is_err());
    }

    #[test]
    fn test_engine_buffer_access() {
        let cfg = CommConfig::new(2).with_send_buffer_capacity(32);
        let e = ParallelCommEngine::new(cfg).unwrap();
        let b = e.buffer(0).unwrap();
        assert_eq!(b.capacity(), 32);
    }

    #[test]
    fn test_engine_buffer_mut_access() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let b = e.buffer_mut(0).unwrap();
        b.write(&[1.0]);
        assert_eq!(b.write_len(), 1);
    }

    #[test]
    fn test_engine_buffer_invalid_worker() {
        let cfg = CommConfig::new(2);
        let e = ParallelCommEngine::new(cfg).unwrap();
        assert!(e.buffer(5).is_err());
    }

    #[test]
    fn test_engine_buffer_mut_invalid_worker() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        assert!(e.buffer_mut(5).is_err());
    }

    #[test]
    fn test_engine_reset_profiler() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        e.all_reduce(&[vec![1.0], vec![2.0]], ReduceOp::Sum).unwrap();
        assert_eq!(e.profiler().event_count(), 1);
        e.reset_profiler();
        assert_eq!(e.profiler().event_count(), 0);
    }

    #[test]
    fn test_engine_profiler_mut() {
        let cfg = CommConfig::new(1);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        e.profiler_mut().record_compute(Duration::from_millis(10));
        assert_eq!(e.profiler().total_compute_time(), Duration::from_millis(10));
    }

    #[test]
    fn test_engine_config_accessor() {
        let cfg = CommConfig::new(3).with_backend(CommBackend::Network);
        let e = ParallelCommEngine::new(cfg).unwrap();
        assert_eq!(e.config().backend(), CommBackend::Network);
    }

    #[test]
    fn test_engine_display() {
        let cfg = CommConfig::new(2);
        let e = ParallelCommEngine::new(cfg).unwrap();
        let s = format!("{e}");
        assert!(s.contains("ParallelCommEngine"));
    }

    #[test]
    fn test_engine_clone() {
        let cfg = CommConfig::new(2);
        let e = ParallelCommEngine::new(cfg).unwrap();
        let e2 = e.clone();
        assert_eq!(e2.num_workers(), 2);
    }

    #[test]
    fn test_engine_multiple_ops_profiled() {
        let cfg = CommConfig::new(2);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        e.all_reduce(&[vec![1.0], vec![2.0]], ReduceOp::Sum).unwrap();
        e.all_gather(&[vec![1.0], vec![2.0]]).unwrap();
        e.broadcast(0, &[1.0]).unwrap();
        assert_eq!(e.profiler().event_count(), 3);
        let summary = e.profiler().per_label_summary();
        assert!(summary.contains_key("allreduce"));
        assert!(summary.contains_key("allgather"));
        assert!(summary.contains_key("broadcast"));
    }

    #[test]
    fn test_engine_single_worker_allreduce() {
        let cfg = CommConfig::new(1);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let r = e.all_reduce(&[vec![42.0]], ReduceOp::Sum).unwrap();
        assert_eq!(r, vec![42.0]);
    }

    #[test]
    fn test_engine_single_worker_broadcast() {
        let cfg = CommConfig::new(1);
        let mut e = ParallelCommEngine::new(cfg).unwrap();
        let r = e.broadcast(0, &[99.0]).unwrap();
        assert_eq!(r, vec![vec![99.0]]);
    }
}
