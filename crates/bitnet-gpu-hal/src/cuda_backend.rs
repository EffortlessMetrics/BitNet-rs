//! Module stub - implementation pending merge from feature branch
//! NVIDIA CUDA backend for GPU compute via cudarc.
//!
//! Provides a CPU-reference HAL abstraction over CUDA device management, stream
//! scheduling, memory allocation, PTX/CUBIN module loading, kernel launch
//! configuration, event-based timing, CUDA graph capture/replay, and multi-GPU
//! peer access.
//!
//! # Structures
//!
//! | Type | Responsibility |
//! |------|----------------|
//! | [`CUDAConfig`] | Device selection and backend feature flags |
//! | [`CUDADevice`] | Physical device properties snapshot |
//! | [`CUDAStream`] | Ordered command queue with priority |
//! | [`CUDAMemory`] | Device / pinned / unified memory regions |
//! | [`CUDAModule`] | Loaded PTX or CUBIN module handle |
//! | [`CUDAKernel`] | Grid/block dims, shared memory, launch |
//! | [`CUDAEvent`] | Timing markers and stream sync points |
//! | [`CUDAGraph`] | Graph capture and replay for reduced launch overhead |
//! | [`CUDAPeerAccess`] | `NVLink` / `PCIe` peer-to-peer access management |
//! | [`CUDABackend`] | Orchestrator: init → select → alloc → load → launch → sync |

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Error ──────────────────────────────────────────────────────────────────

/// Errors produced by the CUDA backend HAL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CUDAError {
    /// Device index out of range.
    InvalidDevice(u32),
    /// Stream has already been destroyed.
    InvalidStream(u64),
    /// Allocation request is zero-sized or exceeds device memory.
    AllocationFailed(String),
    /// PTX/CUBIN failed to load.
    ModuleLoadFailed(String),
    /// Kernel not found in loaded module.
    KernelNotFound(String),
    /// Launch configuration is invalid (zero block dim, etc.).
    InvalidLaunchConfig(String),
    /// Graph capture error.
    GraphError(String),
    /// Peer access could not be enabled.
    PeerAccessFailed { from: u32, to: u32 },
    /// Generic backend error.
    Backend(String),
}

impl fmt::Display for CUDAError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDevice(idx) => write!(f, "invalid device index {idx}"),
            Self::InvalidStream(id) => write!(f, "invalid stream {id}"),
            Self::AllocationFailed(msg) => write!(f, "allocation failed: {msg}"),
            Self::ModuleLoadFailed(msg) => {
                write!(f, "module load failed: {msg}")
            }
            Self::KernelNotFound(name) => {
                write!(f, "kernel '{name}' not found")
            }
            Self::InvalidLaunchConfig(msg) => {
                write!(f, "invalid launch config: {msg}")
            }
            Self::GraphError(msg) => write!(f, "graph error: {msg}"),
            Self::PeerAccessFailed { from, to } => {
                write!(f, "peer access {from} → {to} failed")
            }
            Self::Backend(msg) => write!(f, "backend: {msg}"),
        }
    }
}

impl std::error::Error for CUDAError {}

pub type CUDAResult<T> = Result<T, CUDAError>;

// ── ID generator ───────────────────────────────────────────────────────────

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

// ── CUDAConfig ─────────────────────────────────────────────────────────────

/// Configuration for CUDA backend initialisation.
#[derive(Debug, Clone)]
pub struct CUDAConfig {
    /// Ordinal of the GPU to use (0-based).
    pub device_index: u32,
    /// Number of concurrent CUDA streams to create.
    pub stream_count: u32,
    /// Enable CUDA memory-pool (`cudaMallocAsync`) path.
    pub memory_pool_enabled: bool,
    /// Enable CUDA graph capture for kernel replay.
    pub graph_capture_enabled: bool,
}

impl Default for CUDAConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            stream_count: 2,
            memory_pool_enabled: true,
            graph_capture_enabled: false,
        }
    }
}

impl CUDAConfig {
    /// Create a config targeting a specific device.
    pub fn for_device(device_index: u32) -> Self {
        Self { device_index, ..Self::default() }
    }

    /// Builder – set stream count.
    #[must_use]
    pub const fn with_stream_count(mut self, count: u32) -> Self {
        self.stream_count = count;
        self
    }

    /// Builder – toggle memory pool.
    #[must_use]
    pub const fn with_memory_pool(mut self, enabled: bool) -> Self {
        self.memory_pool_enabled = enabled;
        self
    }

    /// Builder – toggle graph capture.
    #[must_use]
    pub const fn with_graph_capture(mut self, enabled: bool) -> Self {
        self.graph_capture_enabled = enabled;
        self
    }
}

// ── CUDADevice ─────────────────────────────────────────────────────────────

/// Compute-capability pair (major, minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// True when the device supports the requested minimum capability.
    pub fn supports(&self, min_major: u32, min_minor: u32) -> bool {
        (self.major, self.minor) >= (min_major, min_minor)
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sm_{}{}", self.major, self.minor)
    }
}

/// Snapshot of a physical CUDA device's properties.
#[derive(Debug, Clone)]
pub struct CUDADevice {
    /// Device ordinal.
    pub index: u32,
    /// Human-readable device name.
    pub name: String,
    /// Number of streaming multi-processors.
    pub sm_count: u32,
    /// Compute capability.
    pub compute_capability: ComputeCapability,
    /// Total device global memory in bytes.
    pub total_memory_bytes: u64,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block in bytes.
    pub max_shared_memory_per_block: u32,
    /// Warp size (typically 32).
    pub warp_size: u32,
}

impl CUDADevice {
    /// Construct a synthetic device description (CPU reference).
    pub fn synthetic(
        index: u32,
        name: impl Into<String>,
        sm_count: u32,
        cc: ComputeCapability,
        total_memory_bytes: u64,
    ) -> Self {
        Self {
            index,
            name: name.into(),
            sm_count,
            compute_capability: cc,
            total_memory_bytes,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            warp_size: 32,
        }
    }

    /// Available memory (reference impl assumes 90% available).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
    pub fn available_memory_bytes(&self) -> u64 {
        (self.total_memory_bytes as f64 * 0.9) as u64
    }

    /// Supports given minimum compute capability.
    pub fn supports_cc(&self, major: u32, minor: u32) -> bool {
        self.compute_capability.supports(major, minor)
    }
}

impl fmt::Display for CUDADevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPU {} ({}, {} SMs, {}, {} MiB)",
            self.index,
            self.name,
            self.sm_count,
            self.compute_capability,
            self.total_memory_bytes / (1024 * 1024),
        )
    }
}

// ── CUDAStream ─────────────────────────────────────────────────────────────

/// Stream priority levels.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamPriority {
    /// Default scheduling priority.
    #[default]
    Normal,
    /// Higher scheduling priority (lower numeric value in CUDA).
    High,
    /// Lowest scheduling priority.
    Low,
}

/// CUDA stream handle with priority and work tracking.
#[derive(Debug)]
pub struct CUDAStream {
    id: u64,
    device_index: u32,
    priority: StreamPriority,
    work_items: u64,
    synchronized: bool,
}

impl CUDAStream {
    /// Create a new stream on the given device with a priority.
    pub fn new(device_index: u32, priority: StreamPriority) -> Self {
        Self { id: next_id(), device_index, priority, work_items: 0, synchronized: true }
    }

    /// Unique stream id.
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Device this stream belongs to.
    pub const fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Priority.
    pub const fn priority(&self) -> StreamPriority {
        self.priority
    }

    /// Number of work items submitted to this stream.
    pub const fn work_items(&self) -> u64 {
        self.work_items
    }

    /// Record that work was submitted.
    pub const fn record_work(&mut self) {
        self.work_items += 1;
        self.synchronized = false;
    }

    /// Synchronise the stream (CPU reference: no-op, marks synced).
    pub const fn synchronize(&mut self) -> CUDAResult<()> {
        self.synchronized = true;
        Ok(())
    }

    /// Whether the stream has been synchronised since the last work item.
    pub const fn is_synchronized(&self) -> bool {
        self.synchronized
    }
}

impl fmt::Display for CUDAStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stream(id={}, dev={}, pri={:?}, work={})",
            self.id, self.device_index, self.priority, self.work_items,
        )
    }
}

// ── CUDAMemory ─────────────────────────────────────────────────────────────

/// Kind of device memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// Device-local (VRAM).
    Device,
    /// Host-pinned (page-locked).
    Pinned,
    /// Unified (managed, auto-migrated).
    Unified,
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => write!(f, "device"),
            Self::Pinned => write!(f, "pinned"),
            Self::Unified => write!(f, "unified"),
        }
    }
}

/// A memory allocation handle.
#[derive(Debug)]
pub struct CUDAMemory {
    id: u64,
    kind: MemoryKind,
    size_bytes: usize,
    device_index: u32,
    /// CPU-side backing buffer for the reference implementation.
    buffer: Vec<u8>,
}

impl CUDAMemory {
    /// Allocate memory on the given device.
    pub fn allocate(device_index: u32, size_bytes: usize, kind: MemoryKind) -> CUDAResult<Self> {
        if size_bytes == 0 {
            return Err(CUDAError::AllocationFailed("zero-size allocation".into()));
        }
        Ok(Self { id: next_id(), kind, size_bytes, device_index, buffer: vec![0u8; size_bytes] })
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub const fn kind(&self) -> MemoryKind {
        self.kind
    }

    pub const fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    pub const fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Write `data` at `offset` into the allocation.
    pub fn write(&mut self, offset: usize, data: &[u8]) -> CUDAResult<()> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| CUDAError::AllocationFailed("offset overflow".into()))?;
        if end > self.size_bytes {
            return Err(CUDAError::AllocationFailed(format!(
                "write past end: offset={offset} len={} size={}",
                data.len(),
                self.size_bytes,
            )));
        }
        self.buffer[offset..end].copy_from_slice(data);
        Ok(())
    }

    /// Read `len` bytes from `offset`.
    pub fn read(&self, offset: usize, len: usize) -> CUDAResult<&[u8]> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| CUDAError::AllocationFailed("offset overflow".into()))?;
        if end > self.size_bytes {
            return Err(CUDAError::AllocationFailed(format!(
                "read past end: offset={offset} len={len} size={}",
                self.size_bytes,
            )));
        }
        Ok(&self.buffer[offset..end])
    }

    /// Raw immutable view of the entire backing buffer.
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
}

impl fmt::Display for CUDAMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mem(id={}, {}, {} B, dev={})",
            self.id, self.kind, self.size_bytes, self.device_index,
        )
    }
}

// ── CUDAModule ─────────────────────────────────────────────────────────────

/// Source format of a GPU module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleFormat {
    /// Parallel Thread Execution (text-based).
    PTX,
    /// Pre-compiled binary.
    CUBIN,
}

impl fmt::Display for ModuleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PTX => write!(f, "PTX"),
            Self::CUBIN => write!(f, "CUBIN"),
        }
    }
}

/// A loaded PTX or CUBIN module that exposes named kernels.
#[derive(Debug)]
pub struct CUDAModule {
    id: u64,
    name: String,
    format: ModuleFormat,
    kernel_names: Vec<String>,
    source_size_bytes: usize,
}

impl CUDAModule {
    /// Load a module from raw bytes.
    ///
    /// In this CPU reference implementation the bytes are not interpreted; the
    /// caller supplies the list of kernel entry-point names that the module
    /// would expose.
    pub fn load(
        name: impl Into<String>,
        format: ModuleFormat,
        source_bytes: &[u8],
        kernel_names: Vec<String>,
    ) -> CUDAResult<Self> {
        if source_bytes.is_empty() {
            return Err(CUDAError::ModuleLoadFailed("empty source bytes".into()));
        }
        Ok(Self {
            id: next_id(),
            name: name.into(),
            format,
            kernel_names,
            source_size_bytes: source_bytes.len(),
        })
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn format(&self) -> ModuleFormat {
        self.format
    }

    pub fn kernel_names(&self) -> &[String] {
        &self.kernel_names
    }

    pub const fn source_size_bytes(&self) -> usize {
        self.source_size_bytes
    }

    /// True when the module exposes a kernel with the given name.
    pub fn has_kernel(&self, name: &str) -> bool {
        self.kernel_names.iter().any(|k| k == name)
    }
}

impl fmt::Display for CUDAModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Module(id={}, '{}', {}, {} kernels)",
            self.id,
            self.name,
            self.format,
            self.kernel_names.len(),
        )
    }
}

// ── CUDAKernel ─────────────────────────────────────────────────────────────

/// 3-dimensional grid or block size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// 1-D convenience.
    pub const fn x_only(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Total number of elements.
    pub fn volume(&self) -> u64 {
        u64::from(self.x) * u64::from(self.y) * u64::from(self.z)
    }
}

impl fmt::Display for Dim3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// Kernel launch configuration and execution handle.
#[derive(Debug)]
pub struct CUDAKernel {
    module_id: u64,
    function_name: String,
    grid: Dim3,
    block: Dim3,
    shared_memory_bytes: u32,
    launch_count: u64,
}

impl CUDAKernel {
    /// Create a kernel handle from a module and function name.
    pub fn new(module: &CUDAModule, function_name: impl Into<String>) -> CUDAResult<Self> {
        let function_name = function_name.into();
        if !module.has_kernel(&function_name) {
            return Err(CUDAError::KernelNotFound(function_name));
        }
        Ok(Self {
            module_id: module.id(),
            function_name,
            grid: Dim3::x_only(1),
            block: Dim3::x_only(1),
            shared_memory_bytes: 0,
            launch_count: 0,
        })
    }

    pub const fn module_id(&self) -> u64 {
        self.module_id
    }

    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    pub const fn grid(&self) -> Dim3 {
        self.grid
    }

    pub const fn block(&self) -> Dim3 {
        self.block
    }

    pub const fn shared_memory_bytes(&self) -> u32 {
        self.shared_memory_bytes
    }

    pub const fn launch_count(&self) -> u64 {
        self.launch_count
    }

    /// Set grid dimensions.
    pub const fn set_grid(&mut self, grid: Dim3) {
        self.grid = grid;
    }

    /// Set block dimensions.
    pub const fn set_block(&mut self, block: Dim3) {
        self.block = block;
    }

    /// Set dynamic shared memory bytes.
    pub const fn set_shared_memory(&mut self, bytes: u32) {
        self.shared_memory_bytes = bytes;
    }

    /// Validate launch config against device limits.
    pub fn validate(&self, device: &CUDADevice) -> CUDAResult<()> {
        let threads = self.block.volume();
        if threads == 0 {
            return Err(CUDAError::InvalidLaunchConfig("block volume is zero".into()));
        }
        if threads > u64::from(device.max_threads_per_block) {
            return Err(CUDAError::InvalidLaunchConfig(format!(
                "block has {threads} threads, max {}",
                device.max_threads_per_block,
            )));
        }
        if self.shared_memory_bytes > device.max_shared_memory_per_block {
            return Err(CUDAError::InvalidLaunchConfig(format!(
                "shared memory {} > max {}",
                self.shared_memory_bytes, device.max_shared_memory_per_block,
            )));
        }
        if self.grid.volume() == 0 {
            return Err(CUDAError::InvalidLaunchConfig("grid volume is zero".into()));
        }
        Ok(())
    }

    /// Launch the kernel on the given stream (CPU ref: validates and counts).
    pub fn launch(&mut self, stream: &mut CUDAStream, device: &CUDADevice) -> CUDAResult<()> {
        self.validate(device)?;
        stream.record_work();
        self.launch_count += 1;
        Ok(())
    }
}

impl fmt::Display for CUDAKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Kernel('{}', grid={}, block={}, smem={})",
            self.function_name, self.grid, self.block, self.shared_memory_bytes,
        )
    }
}

// ── CUDAEvent ──────────────────────────────────────────────────────────────

/// A CUDA event used for timing and synchronisation.
#[derive(Debug)]
pub struct CUDAEvent {
    id: u64,
    recorded_at: Option<Instant>,
    label: String,
}

impl CUDAEvent {
    /// Create a new unrecorded event with an optional label.
    pub fn new(label: impl Into<String>) -> Self {
        Self { id: next_id(), recorded_at: None, label: label.into() }
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// Record the event on a stream (CPU ref: captures wall-clock time).
    pub fn record(&mut self, stream: &mut CUDAStream) -> CUDAResult<()> {
        stream.record_work();
        self.recorded_at = Some(Instant::now());
        Ok(())
    }

    /// True when `record` has been called at least once.
    pub const fn is_recorded(&self) -> bool {
        self.recorded_at.is_some()
    }

    /// Elapsed time between two recorded events.
    pub fn elapsed_since(&self, earlier: &Self) -> CUDAResult<Duration> {
        match (earlier.recorded_at, self.recorded_at) {
            (Some(start), Some(end)) => Ok(end.duration_since(start)),
            _ => Err(CUDAError::Backend("both events must be recorded".into())),
        }
    }

    /// Synchronise on this event (CPU ref: no-op).
    pub fn synchronize(&self) -> CUDAResult<()> {
        if self.recorded_at.is_none() {
            return Err(CUDAError::Backend("cannot synchronise unrecorded event".into()));
        }
        Ok(())
    }
}

impl fmt::Display for CUDAEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.recorded_at.is_some() { "recorded" } else { "pending" };
        write!(f, "Event(id={}, '{}', {})", self.id, self.label, state)
    }
}

// ── CUDAGraph ──────────────────────────────────────────────────────────────

/// State of a CUDA graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphState {
    /// Ready for capture.
    Idle,
    /// Currently recording operations.
    Capturing,
    /// Capture complete, graph is instantiated and ready to replay.
    Ready,
}

impl fmt::Display for GraphState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Capturing => write!(f, "capturing"),
            Self::Ready => write!(f, "ready"),
        }
    }
}

/// CUDA graph capture and replay for reduced kernel launch overhead.
#[derive(Debug)]
pub struct CUDAGraph {
    id: u64,
    label: String,
    state: GraphState,
    captured_ops: u64,
    replay_count: u64,
}

impl CUDAGraph {
    /// Create a new idle graph.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            id: next_id(),
            label: label.into(),
            state: GraphState::Idle,
            captured_ops: 0,
            replay_count: 0,
        }
    }

    pub const fn id(&self) -> u64 {
        self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn state(&self) -> GraphState {
        self.state
    }

    pub const fn captured_ops(&self) -> u64 {
        self.captured_ops
    }

    pub const fn replay_count(&self) -> u64 {
        self.replay_count
    }

    /// Begin capture. Fails if not idle.
    pub fn begin_capture(&mut self) -> CUDAResult<()> {
        if self.state != GraphState::Idle {
            return Err(CUDAError::GraphError(format!(
                "cannot begin capture in state {}",
                self.state,
            )));
        }
        self.state = GraphState::Capturing;
        self.captured_ops = 0;
        Ok(())
    }

    /// Record an operation during capture.
    pub fn record_op(&mut self) -> CUDAResult<()> {
        if self.state != GraphState::Capturing {
            return Err(CUDAError::GraphError("not in capturing state".into()));
        }
        self.captured_ops += 1;
        Ok(())
    }

    /// End capture and instantiate the graph.
    pub fn end_capture(&mut self) -> CUDAResult<()> {
        if self.state != GraphState::Capturing {
            return Err(CUDAError::GraphError("not in capturing state".into()));
        }
        if self.captured_ops == 0 {
            return Err(CUDAError::GraphError("cannot instantiate empty graph".into()));
        }
        self.state = GraphState::Ready;
        Ok(())
    }

    /// Replay the instantiated graph on a stream.
    pub fn replay(&mut self, stream: &mut CUDAStream) -> CUDAResult<()> {
        if self.state != GraphState::Ready {
            return Err(CUDAError::GraphError(format!("cannot replay in state {}", self.state,)));
        }
        stream.record_work();
        self.replay_count += 1;
        Ok(())
    }

    /// Reset back to idle so the graph can be re-captured.
    pub const fn reset(&mut self) {
        self.state = GraphState::Idle;
        self.captured_ops = 0;
        self.replay_count = 0;
    }
}

impl fmt::Display for CUDAGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Graph(id={}, '{}', {}, ops={}, replays={})",
            self.id, self.label, self.state, self.captured_ops, self.replay_count,
        )
    }
}

// ── CUDAPeerAccess ─────────────────────────────────────────────────────────

/// Tracks `NVLink` / `PCIe` peer-to-peer access between devices.
#[derive(Debug)]
pub struct CUDAPeerAccess {
    /// Set of enabled `(from, to)` device pairs.
    enabled_pairs: Vec<(u32, u32)>,
}

impl CUDAPeerAccess {
    pub const fn new() -> Self {
        Self { enabled_pairs: Vec::new() }
    }

    /// Enable peer access from one device to another.
    pub fn enable(&mut self, from: u32, to: u32) -> CUDAResult<()> {
        if from == to {
            return Err(CUDAError::PeerAccessFailed { from, to });
        }
        if self.is_enabled(from, to) {
            return Ok(());
        }
        self.enabled_pairs.push((from, to));
        Ok(())
    }

    /// Disable peer access.
    pub fn disable(&mut self, from: u32, to: u32) {
        self.enabled_pairs.retain(|&pair| pair != (from, to));
    }

    /// Check whether peer access is currently enabled.
    pub fn is_enabled(&self, from: u32, to: u32) -> bool {
        self.enabled_pairs.contains(&(from, to))
    }

    /// Return all currently-enabled device pairs.
    pub fn enabled_pairs(&self) -> &[(u32, u32)] {
        &self.enabled_pairs
    }

    /// Number of active peer links.
    pub const fn active_count(&self) -> usize {
        self.enabled_pairs.len()
    }

    /// Check if a device can access another (reference: always yes for
    /// different devices).
    pub const fn can_access(_from: u32, _to: u32) -> bool {
        true
    }
}

impl Default for CUDAPeerAccess {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CUDAPeerAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PeerAccess({} links)", self.enabled_pairs.len())
    }
}

// ── CUDABackend ────────────────────────────────────────────────────────────

/// Backend state tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendState {
    /// Not yet initialised.
    Uninitialised,
    /// Initialised and ready.
    Ready,
    /// Shut down.
    Shutdown,
}

impl fmt::Display for BackendState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninitialised => write!(f, "uninitialised"),
            Self::Ready => write!(f, "ready"),
            Self::Shutdown => write!(f, "shutdown"),
        }
    }
}

/// Top-level CUDA backend orchestrator.
///
/// Lifecycle: `init` → device select → alloc → load → launch → sync →
/// `shutdown`.
#[derive(Debug)]
pub struct CUDABackend {
    config: CUDAConfig,
    state: BackendState,
    device: Option<CUDADevice>,
    streams: Vec<CUDAStream>,
    allocations: HashMap<u64, CUDAMemory>,
    modules: HashMap<u64, CUDAModule>,
    peer_access: CUDAPeerAccess,
    total_allocated_bytes: usize,
}

impl CUDABackend {
    /// Create an uninitialised backend with the given config.
    pub fn new(config: CUDAConfig) -> Self {
        Self {
            config,
            state: BackendState::Uninitialised,
            device: None,
            streams: Vec::new(),
            allocations: HashMap::new(),
            modules: HashMap::new(),
            peer_access: CUDAPeerAccess::new(),
            total_allocated_bytes: 0,
        }
    }

    /// Initialise the backend: select device and create streams.
    pub fn init(&mut self, available_devices: &[CUDADevice]) -> CUDAResult<()> {
        if self.state != BackendState::Uninitialised {
            return Err(CUDAError::Backend(format!("cannot init in state {}", self.state,)));
        }
        let dev = available_devices
            .iter()
            .find(|d| d.index == self.config.device_index)
            .ok_or(CUDAError::InvalidDevice(self.config.device_index))?;

        self.device = Some(dev.clone());

        for i in 0..self.config.stream_count {
            let priority = if i == 0 { StreamPriority::High } else { StreamPriority::Normal };
            self.streams.push(CUDAStream::new(self.config.device_index, priority));
        }

        self.state = BackendState::Ready;
        log::info!("CUDA backend initialised on {dev}");
        Ok(())
    }

    pub const fn state(&self) -> BackendState {
        self.state
    }

    pub const fn config(&self) -> &CUDAConfig {
        &self.config
    }

    pub const fn device(&self) -> Option<&CUDADevice> {
        self.device.as_ref()
    }

    pub fn streams(&self) -> &[CUDAStream] {
        &self.streams
    }

    pub fn stream_mut(&mut self, index: usize) -> CUDAResult<&mut CUDAStream> {
        self.streams.get_mut(index).ok_or(CUDAError::InvalidStream(index as u64))
    }

    /// Allocate memory on the active device.
    pub fn allocate(&mut self, size_bytes: usize, kind: MemoryKind) -> CUDAResult<u64> {
        self.require_ready()?;
        let dev = self.device.as_ref().unwrap();
        if size_bytes as u64 > dev.available_memory_bytes() {
            return Err(CUDAError::AllocationFailed("exceeds device memory".into()));
        }
        let mem = CUDAMemory::allocate(self.config.device_index, size_bytes, kind)?;
        let id = mem.id();
        self.total_allocated_bytes += size_bytes;
        self.allocations.insert(id, mem);
        Ok(id)
    }

    /// Retrieve an allocation by id.
    pub fn get_allocation(&self, id: u64) -> CUDAResult<&CUDAMemory> {
        self.allocations
            .get(&id)
            .ok_or_else(|| CUDAError::AllocationFailed(format!("id {id} not found")))
    }

    /// Retrieve a mutable allocation by id.
    pub fn get_allocation_mut(&mut self, id: u64) -> CUDAResult<&mut CUDAMemory> {
        self.allocations
            .get_mut(&id)
            .ok_or_else(|| CUDAError::AllocationFailed(format!("id {id} not found")))
    }

    /// Free an allocation.
    pub fn free(&mut self, id: u64) -> CUDAResult<()> {
        let mem = self
            .allocations
            .remove(&id)
            .ok_or_else(|| CUDAError::AllocationFailed(format!("id {id} not found")))?;
        self.total_allocated_bytes = self.total_allocated_bytes.saturating_sub(mem.size_bytes());
        Ok(())
    }

    /// Total bytes currently allocated.
    pub const fn total_allocated_bytes(&self) -> usize {
        self.total_allocated_bytes
    }

    /// Load a module into the backend.
    pub fn load_module(
        &mut self,
        name: impl Into<String>,
        format: ModuleFormat,
        source_bytes: &[u8],
        kernel_names: Vec<String>,
    ) -> CUDAResult<u64> {
        self.require_ready()?;
        let module = CUDAModule::load(name, format, source_bytes, kernel_names)?;
        let id = module.id();
        self.modules.insert(id, module);
        Ok(id)
    }

    /// Retrieve a loaded module by id.
    pub fn get_module(&self, id: u64) -> CUDAResult<&CUDAModule> {
        self.modules
            .get(&id)
            .ok_or_else(|| CUDAError::ModuleLoadFailed(format!("id {id} not found")))
    }

    /// Unload a module.
    pub fn unload_module(&mut self, id: u64) -> CUDAResult<()> {
        self.modules
            .remove(&id)
            .ok_or_else(|| CUDAError::ModuleLoadFailed(format!("id {id} not found")))?;
        Ok(())
    }

    /// Access peer-access manager.
    pub const fn peer_access(&self) -> &CUDAPeerAccess {
        &self.peer_access
    }

    /// Access peer-access manager mutably.
    pub const fn peer_access_mut(&mut self) -> &mut CUDAPeerAccess {
        &mut self.peer_access
    }

    /// Synchronise all streams.
    pub fn synchronize_all(&mut self) -> CUDAResult<()> {
        self.require_ready()?;
        for stream in &mut self.streams {
            stream.synchronize()?;
        }
        Ok(())
    }

    /// Shut down the backend, releasing all resources.
    pub fn shutdown(&mut self) -> CUDAResult<()> {
        self.require_ready()?;
        self.synchronize_all()?;
        self.allocations.clear();
        self.modules.clear();
        self.streams.clear();
        self.total_allocated_bytes = 0;
        self.state = BackendState::Shutdown;
        log::info!("CUDA backend shut down");
        Ok(())
    }

    /// Internal guard: ensure backend is `Ready`.
    fn require_ready(&self) -> CUDAResult<()> {
        if self.state != BackendState::Ready {
            return Err(CUDAError::Backend(format!("backend not ready (state={})", self.state,)));
        }
        Ok(())
    }
}

impl fmt::Display for CUDABackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dev_name = self.device.as_ref().map_or("none", |d| d.name.as_str());
        write!(
            f,
            "CUDABackend(state={}, dev='{}', streams={}, allocs={}, mods={})",
            self.state,
            dev_name,
            self.streams.len(),
            self.allocations.len(),
            self.modules.len(),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn test_device() -> CUDADevice {
        CUDADevice::synthetic(
            0,
            "Test GPU",
            80,
            ComputeCapability::new(8, 6),
            16 * 1024 * 1024 * 1024, // 16 GiB
        )
    }

    fn test_devices() -> Vec<CUDADevice> {
        vec![
            test_device(),
            CUDADevice::synthetic(
                1,
                "Test GPU 1",
                108,
                ComputeCapability::new(9, 0),
                24 * 1024 * 1024 * 1024,
            ),
        ]
    }

    fn test_module() -> CUDAModule {
        CUDAModule::load(
            "test_mod",
            ModuleFormat::PTX,
            b".version 7.0\n",
            vec!["matmul".into(), "relu".into()],
        )
        .unwrap()
    }

    fn ready_backend() -> CUDABackend {
        let mut backend = CUDABackend::new(CUDAConfig::default());
        backend.init(&test_devices()).unwrap();
        backend
    }

    // ── CUDAConfig tests ──────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = CUDAConfig::default();
        assert_eq!(cfg.device_index, 0);
        assert_eq!(cfg.stream_count, 2);
        assert!(cfg.memory_pool_enabled);
        assert!(!cfg.graph_capture_enabled);
    }

    #[test]
    fn config_for_device() {
        let cfg = CUDAConfig::for_device(3);
        assert_eq!(cfg.device_index, 3);
        assert_eq!(cfg.stream_count, 2);
    }

    #[test]
    fn config_builder_stream_count() {
        let cfg = CUDAConfig::default().with_stream_count(8);
        assert_eq!(cfg.stream_count, 8);
    }

    #[test]
    fn config_builder_memory_pool() {
        let cfg = CUDAConfig::default().with_memory_pool(false);
        assert!(!cfg.memory_pool_enabled);
    }

    #[test]
    fn config_builder_graph_capture() {
        let cfg = CUDAConfig::default().with_graph_capture(true);
        assert!(cfg.graph_capture_enabled);
    }

    #[test]
    fn config_builder_chain() {
        let cfg = CUDAConfig::for_device(1)
            .with_stream_count(4)
            .with_memory_pool(false)
            .with_graph_capture(true);
        assert_eq!(cfg.device_index, 1);
        assert_eq!(cfg.stream_count, 4);
        assert!(!cfg.memory_pool_enabled);
        assert!(cfg.graph_capture_enabled);
    }

    // ── ComputeCapability tests ───────────────────────────────────────

    #[test]
    fn cc_supports_exact() {
        let cc = ComputeCapability::new(8, 6);
        assert!(cc.supports(8, 6));
    }

    #[test]
    fn cc_supports_lower() {
        let cc = ComputeCapability::new(9, 0);
        assert!(cc.supports(8, 6));
    }

    #[test]
    fn cc_does_not_support_higher() {
        let cc = ComputeCapability::new(7, 5);
        assert!(!cc.supports(8, 0));
    }

    #[test]
    fn cc_display() {
        let cc = ComputeCapability::new(8, 9);
        assert_eq!(cc.to_string(), "sm_89");
    }

    // ── CUDADevice tests ──────────────────────────────────────────────

    #[test]
    fn device_synthetic_defaults() {
        let dev = test_device();
        assert_eq!(dev.index, 0);
        assert_eq!(dev.sm_count, 80);
        assert_eq!(dev.max_threads_per_block, 1024);
        assert_eq!(dev.warp_size, 32);
    }

    #[test]
    fn device_available_memory() {
        let dev = test_device();
        assert!(dev.available_memory_bytes() < dev.total_memory_bytes);
        assert!(dev.available_memory_bytes() > 0);
    }

    #[test]
    fn device_supports_cc() {
        let dev = test_device();
        assert!(dev.supports_cc(8, 0));
        assert!(!dev.supports_cc(9, 0));
    }

    #[test]
    fn device_display() {
        let dev = test_device();
        let s = dev.to_string();
        assert!(s.contains("Test GPU"));
        assert!(s.contains("80 SMs"));
    }

    // ── CUDAStream tests ─────────────────────────────────────────────

    #[test]
    fn stream_initial_state() {
        let s = CUDAStream::new(0, StreamPriority::Normal);
        assert_eq!(s.device_index(), 0);
        assert_eq!(s.priority(), StreamPriority::Normal);
        assert_eq!(s.work_items(), 0);
        assert!(s.is_synchronized());
    }

    #[test]
    fn stream_record_work_marks_unsynchronized() {
        let mut s = CUDAStream::new(0, StreamPriority::High);
        s.record_work();
        assert_eq!(s.work_items(), 1);
        assert!(!s.is_synchronized());
    }

    #[test]
    fn stream_synchronize_resets_flag() {
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        s.record_work();
        s.synchronize().unwrap();
        assert!(s.is_synchronized());
    }

    #[test]
    fn stream_multiple_work_items() {
        let mut s = CUDAStream::new(0, StreamPriority::Low);
        for _ in 0..10 {
            s.record_work();
        }
        assert_eq!(s.work_items(), 10);
    }

    #[test]
    fn stream_unique_ids() {
        let a = CUDAStream::new(0, StreamPriority::Normal);
        let b = CUDAStream::new(0, StreamPriority::Normal);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn stream_display() {
        let s = CUDAStream::new(1, StreamPriority::High);
        let out = s.to_string();
        assert!(out.contains("Stream"));
        assert!(out.contains("dev=1"));
    }

    #[test]
    fn stream_priority_default() {
        assert_eq!(StreamPriority::default(), StreamPriority::Normal);
    }

    // ── CUDAMemory tests ─────────────────────────────────────────────

    #[test]
    fn memory_allocate_device() {
        let mem = CUDAMemory::allocate(0, 1024, MemoryKind::Device).unwrap();
        assert_eq!(mem.size_bytes(), 1024);
        assert_eq!(mem.kind(), MemoryKind::Device);
        assert_eq!(mem.device_index(), 0);
    }

    #[test]
    fn memory_allocate_pinned() {
        let mem = CUDAMemory::allocate(0, 256, MemoryKind::Pinned).unwrap();
        assert_eq!(mem.kind(), MemoryKind::Pinned);
    }

    #[test]
    fn memory_allocate_unified() {
        let mem = CUDAMemory::allocate(0, 512, MemoryKind::Unified).unwrap();
        assert_eq!(mem.kind(), MemoryKind::Unified);
    }

    #[test]
    fn memory_zero_size_fails() {
        let err = CUDAMemory::allocate(0, 0, MemoryKind::Device);
        assert!(err.is_err());
    }

    #[test]
    fn memory_write_read_roundtrip() {
        let mut mem = CUDAMemory::allocate(0, 64, MemoryKind::Device).unwrap();
        let data = b"hello CUDA";
        mem.write(0, data).unwrap();
        let back = mem.read(0, data.len()).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn memory_write_at_offset() {
        let mut mem = CUDAMemory::allocate(0, 64, MemoryKind::Device).unwrap();
        mem.write(10, &[0xAB, 0xCD]).unwrap();
        assert_eq!(mem.read(10, 2).unwrap(), &[0xAB, 0xCD]);
    }

    #[test]
    fn memory_write_past_end_fails() {
        let mut mem = CUDAMemory::allocate(0, 8, MemoryKind::Device).unwrap();
        let err = mem.write(4, &[0; 8]);
        assert!(err.is_err());
    }

    #[test]
    fn memory_read_past_end_fails() {
        let mem = CUDAMemory::allocate(0, 8, MemoryKind::Device).unwrap();
        assert!(mem.read(0, 16).is_err());
    }

    #[test]
    fn memory_as_slice() {
        let mem = CUDAMemory::allocate(0, 4, MemoryKind::Device).unwrap();
        assert_eq!(mem.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn memory_display() {
        let mem = CUDAMemory::allocate(0, 100, MemoryKind::Device).unwrap();
        let s = mem.to_string();
        assert!(s.contains("device"));
        assert!(s.contains("100 B"));
    }

    #[test]
    fn memory_kind_display() {
        assert_eq!(MemoryKind::Device.to_string(), "device");
        assert_eq!(MemoryKind::Pinned.to_string(), "pinned");
        assert_eq!(MemoryKind::Unified.to_string(), "unified");
    }

    #[test]
    fn memory_unique_ids() {
        let a = CUDAMemory::allocate(0, 8, MemoryKind::Device).unwrap();
        let b = CUDAMemory::allocate(0, 8, MemoryKind::Device).unwrap();
        assert_ne!(a.id(), b.id());
    }

    // ── CUDAModule tests ─────────────────────────────────────────────

    #[test]
    fn module_load_ptx() {
        let m = CUDAModule::load("my_mod", ModuleFormat::PTX, b".version 7.0", vec!["k1".into()])
            .unwrap();
        assert_eq!(m.name(), "my_mod");
        assert_eq!(m.format(), ModuleFormat::PTX);
        assert_eq!(m.kernel_names().len(), 1);
    }

    #[test]
    fn module_load_cubin() {
        let m = CUDAModule::load(
            "bin",
            ModuleFormat::CUBIN,
            &[0xDE, 0xAD],
            vec!["k1".into(), "k2".into()],
        )
        .unwrap();
        assert_eq!(m.format(), ModuleFormat::CUBIN);
        assert_eq!(m.kernel_names().len(), 2);
    }

    #[test]
    fn module_empty_source_fails() {
        let err = CUDAModule::load("x", ModuleFormat::PTX, b"", vec![]);
        assert!(err.is_err());
    }

    #[test]
    fn module_has_kernel() {
        let m = test_module();
        assert!(m.has_kernel("matmul"));
        assert!(m.has_kernel("relu"));
        assert!(!m.has_kernel("softmax"));
    }

    #[test]
    fn module_source_size() {
        let m = test_module();
        assert!(m.source_size_bytes() > 0);
    }

    #[test]
    fn module_display() {
        let m = test_module();
        let s = m.to_string();
        assert!(s.contains("test_mod"));
        assert!(s.contains("2 kernels"));
    }

    #[test]
    fn module_format_display() {
        assert_eq!(ModuleFormat::PTX.to_string(), "PTX");
        assert_eq!(ModuleFormat::CUBIN.to_string(), "CUBIN");
    }

    // ── Dim3 tests ────────────────────────────────────────────────────

    #[test]
    fn dim3_volume_3d() {
        let d = Dim3::new(4, 8, 2);
        assert_eq!(d.volume(), 64);
    }

    #[test]
    fn dim3_x_only() {
        let d = Dim3::x_only(256);
        assert_eq!(d.volume(), 256);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
    }

    #[test]
    fn dim3_display() {
        assert_eq!(Dim3::new(1, 2, 3).to_string(), "(1, 2, 3)");
    }

    // ── CUDAKernel tests ─────────────────────────────────────────────

    #[test]
    fn kernel_create_valid() {
        let m = test_module();
        let k = CUDAKernel::new(&m, "matmul").unwrap();
        assert_eq!(k.function_name(), "matmul");
        assert_eq!(k.launch_count(), 0);
    }

    #[test]
    fn kernel_create_missing_function() {
        let m = test_module();
        let err = CUDAKernel::new(&m, "nonexistent");
        assert!(matches!(err, Err(CUDAError::KernelNotFound(_))));
    }

    #[test]
    fn kernel_set_grid_block() {
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_grid(Dim3::x_only(128));
        k.set_block(Dim3::x_only(256));
        assert_eq!(k.grid().x, 128);
        assert_eq!(k.block().x, 256);
    }

    #[test]
    fn kernel_set_shared_memory() {
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_shared_memory(16384);
        assert_eq!(k.shared_memory_bytes(), 16384);
    }

    #[test]
    fn kernel_validate_ok() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_grid(Dim3::x_only(64));
        k.set_block(Dim3::x_only(256));
        assert!(k.validate(&dev).is_ok());
    }

    #[test]
    fn kernel_validate_zero_block() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_block(Dim3::new(0, 0, 0));
        assert!(k.validate(&dev).is_err());
    }

    #[test]
    fn kernel_validate_too_many_threads() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_block(Dim3::x_only(2048));
        assert!(matches!(k.validate(&dev), Err(CUDAError::InvalidLaunchConfig(_))));
    }

    #[test]
    fn kernel_validate_shared_memory_exceeded() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_shared_memory(100_000);
        assert!(k.validate(&dev).is_err());
    }

    #[test]
    fn kernel_validate_zero_grid() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_grid(Dim3::new(0, 1, 1));
        assert!(k.validate(&dev).is_err());
    }

    #[test]
    fn kernel_launch_increments_count() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_grid(Dim3::x_only(1));
        k.set_block(Dim3::x_only(32));
        let mut stream = CUDAStream::new(0, StreamPriority::Normal);
        k.launch(&mut stream, &dev).unwrap();
        assert_eq!(k.launch_count(), 1);
        assert_eq!(stream.work_items(), 1);
    }

    #[test]
    fn kernel_launch_multiple() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "relu").unwrap();
        k.set_grid(Dim3::x_only(10));
        k.set_block(Dim3::x_only(128));
        let mut stream = CUDAStream::new(0, StreamPriority::Normal);
        for _ in 0..5 {
            k.launch(&mut stream, &dev).unwrap();
        }
        assert_eq!(k.launch_count(), 5);
        assert_eq!(stream.work_items(), 5);
    }

    #[test]
    fn kernel_launch_invalid_fails() {
        let dev = test_device();
        let m = test_module();
        let mut k = CUDAKernel::new(&m, "matmul").unwrap();
        k.set_block(Dim3::new(0, 0, 0));
        let mut stream = CUDAStream::new(0, StreamPriority::Normal);
        assert!(k.launch(&mut stream, &dev).is_err());
        assert_eq!(k.launch_count(), 0);
    }

    #[test]
    fn kernel_display() {
        let m = test_module();
        let k = CUDAKernel::new(&m, "matmul").unwrap();
        let s = k.to_string();
        assert!(s.contains("matmul"));
    }

    #[test]
    fn kernel_module_id_matches() {
        let m = test_module();
        let k = CUDAKernel::new(&m, "matmul").unwrap();
        assert_eq!(k.module_id(), m.id());
    }

    // ── CUDAEvent tests ──────────────────────────────────────────────

    #[test]
    fn event_new_unrecorded() {
        let e = CUDAEvent::new("start");
        assert!(!e.is_recorded());
        assert_eq!(e.label(), "start");
    }

    #[test]
    fn event_record() {
        let mut e = CUDAEvent::new("start");
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        e.record(&mut s).unwrap();
        assert!(e.is_recorded());
    }

    #[test]
    fn event_elapsed_since() {
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        let mut start = CUDAEvent::new("start");
        start.record(&mut s).unwrap();
        std::thread::sleep(Duration::from_millis(1));
        let mut end = CUDAEvent::new("end");
        end.record(&mut s).unwrap();
        let dt = end.elapsed_since(&start).unwrap();
        assert!(dt >= Duration::from_millis(1));
    }

    #[test]
    fn event_elapsed_unrecorded_fails() {
        let a = CUDAEvent::new("a");
        let b = CUDAEvent::new("b");
        assert!(b.elapsed_since(&a).is_err());
    }

    #[test]
    fn event_synchronize_unrecorded_fails() {
        let e = CUDAEvent::new("x");
        assert!(e.synchronize().is_err());
    }

    #[test]
    fn event_synchronize_recorded_ok() {
        let mut e = CUDAEvent::new("x");
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        e.record(&mut s).unwrap();
        assert!(e.synchronize().is_ok());
    }

    #[test]
    fn event_unique_ids() {
        let a = CUDAEvent::new("a");
        let b = CUDAEvent::new("b");
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn event_display_pending() {
        let e = CUDAEvent::new("my_event");
        let s = e.to_string();
        assert!(s.contains("pending"));
        assert!(s.contains("my_event"));
    }

    #[test]
    fn event_display_recorded() {
        let mut e = CUDAEvent::new("ev");
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        e.record(&mut s).unwrap();
        assert!(e.to_string().contains("recorded"));
    }

    // ── CUDAGraph tests ──────────────────────────────────────────────

    #[test]
    fn graph_initial_state() {
        let g = CUDAGraph::new("g1");
        assert_eq!(g.state(), GraphState::Idle);
        assert_eq!(g.captured_ops(), 0);
        assert_eq!(g.replay_count(), 0);
    }

    #[test]
    fn graph_capture_lifecycle() {
        let mut g = CUDAGraph::new("g");
        g.begin_capture().unwrap();
        assert_eq!(g.state(), GraphState::Capturing);
        g.record_op().unwrap();
        g.record_op().unwrap();
        g.end_capture().unwrap();
        assert_eq!(g.state(), GraphState::Ready);
        assert_eq!(g.captured_ops(), 2);
    }

    #[test]
    fn graph_replay() {
        let mut g = CUDAGraph::new("g");
        g.begin_capture().unwrap();
        g.record_op().unwrap();
        g.end_capture().unwrap();
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        g.replay(&mut s).unwrap();
        g.replay(&mut s).unwrap();
        assert_eq!(g.replay_count(), 2);
        assert_eq!(s.work_items(), 2);
    }

    #[test]
    fn graph_begin_capture_not_idle_fails() {
        let mut g = CUDAGraph::new("g");
        g.begin_capture().unwrap();
        assert!(g.begin_capture().is_err());
    }

    #[test]
    fn graph_record_op_not_capturing_fails() {
        let mut g = CUDAGraph::new("g");
        assert!(g.record_op().is_err());
    }

    #[test]
    fn graph_end_capture_empty_fails() {
        let mut g = CUDAGraph::new("g");
        g.begin_capture().unwrap();
        assert!(g.end_capture().is_err());
    }

    #[test]
    fn graph_replay_not_ready_fails() {
        let mut g = CUDAGraph::new("g");
        let mut s = CUDAStream::new(0, StreamPriority::Normal);
        assert!(g.replay(&mut s).is_err());
    }

    #[test]
    fn graph_reset() {
        let mut g = CUDAGraph::new("g");
        g.begin_capture().unwrap();
        g.record_op().unwrap();
        g.end_capture().unwrap();
        g.reset();
        assert_eq!(g.state(), GraphState::Idle);
        assert_eq!(g.captured_ops(), 0);
        assert_eq!(g.replay_count(), 0);
    }

    #[test]
    fn graph_recapture_after_reset() {
        let mut g = CUDAGraph::new("g");
        g.begin_capture().unwrap();
        g.record_op().unwrap();
        g.end_capture().unwrap();
        g.reset();
        g.begin_capture().unwrap();
        g.record_op().unwrap();
        g.record_op().unwrap();
        g.record_op().unwrap();
        g.end_capture().unwrap();
        assert_eq!(g.captured_ops(), 3);
    }

    #[test]
    fn graph_display() {
        let g = CUDAGraph::new("my_graph");
        let s = g.to_string();
        assert!(s.contains("my_graph"));
        assert!(s.contains("idle"));
    }

    #[test]
    fn graph_state_display() {
        assert_eq!(GraphState::Idle.to_string(), "idle");
        assert_eq!(GraphState::Capturing.to_string(), "capturing");
        assert_eq!(GraphState::Ready.to_string(), "ready");
    }

    // ── CUDAPeerAccess tests ─────────────────────────────────────────

    #[test]
    fn peer_access_default_empty() {
        let p = CUDAPeerAccess::new();
        assert_eq!(p.active_count(), 0);
    }

    #[test]
    fn peer_access_enable() {
        let mut p = CUDAPeerAccess::new();
        p.enable(0, 1).unwrap();
        assert!(p.is_enabled(0, 1));
        assert!(!p.is_enabled(1, 0));
    }

    #[test]
    fn peer_access_enable_self_fails() {
        let mut p = CUDAPeerAccess::new();
        assert!(p.enable(0, 0).is_err());
    }

    #[test]
    fn peer_access_enable_idempotent() {
        let mut p = CUDAPeerAccess::new();
        p.enable(0, 1).unwrap();
        p.enable(0, 1).unwrap();
        assert_eq!(p.active_count(), 1);
    }

    #[test]
    fn peer_access_disable() {
        let mut p = CUDAPeerAccess::new();
        p.enable(0, 1).unwrap();
        p.disable(0, 1);
        assert!(!p.is_enabled(0, 1));
        assert_eq!(p.active_count(), 0);
    }

    #[test]
    fn peer_access_enabled_pairs() {
        let mut p = CUDAPeerAccess::new();
        p.enable(0, 1).unwrap();
        p.enable(1, 0).unwrap();
        assert_eq!(p.enabled_pairs().len(), 2);
    }

    #[test]
    fn peer_access_can_access() {
        assert!(CUDAPeerAccess::can_access(0, 1));
    }

    #[test]
    fn peer_access_display() {
        let p = CUDAPeerAccess::new();
        assert!(p.to_string().contains("0 links"));
    }

    // ── CUDABackend tests ────────────────────────────────────────────

    #[test]
    fn backend_new_uninitialised() {
        let b = CUDABackend::new(CUDAConfig::default());
        assert_eq!(b.state(), BackendState::Uninitialised);
        assert!(b.device().is_none());
    }

    #[test]
    fn backend_init_success() {
        let b = ready_backend();
        assert_eq!(b.state(), BackendState::Ready);
        assert!(b.device().is_some());
        assert_eq!(b.streams().len(), 2);
    }

    #[test]
    fn backend_init_invalid_device() {
        let mut b = CUDABackend::new(CUDAConfig::for_device(99));
        let err = b.init(&test_devices());
        assert!(matches!(err, Err(CUDAError::InvalidDevice(99))));
    }

    #[test]
    fn backend_init_double_fails() {
        let mut b = ready_backend();
        assert!(b.init(&test_devices()).is_err());
    }

    #[test]
    fn backend_first_stream_high_priority() {
        let b = ready_backend();
        assert_eq!(b.streams()[0].priority(), StreamPriority::High);
        assert_eq!(b.streams()[1].priority(), StreamPriority::Normal);
    }

    #[test]
    fn backend_allocate_and_retrieve() {
        let mut b = ready_backend();
        let id = b.allocate(1024, MemoryKind::Device).unwrap();
        let mem = b.get_allocation(id).unwrap();
        assert_eq!(mem.size_bytes(), 1024);
    }

    #[test]
    fn backend_allocate_tracks_total() {
        let mut b = ready_backend();
        b.allocate(100, MemoryKind::Device).unwrap();
        b.allocate(200, MemoryKind::Pinned).unwrap();
        assert_eq!(b.total_allocated_bytes(), 300);
    }

    #[test]
    fn backend_free_reduces_total() {
        let mut b = ready_backend();
        let id = b.allocate(512, MemoryKind::Device).unwrap();
        b.free(id).unwrap();
        assert_eq!(b.total_allocated_bytes(), 0);
    }

    #[test]
    fn backend_free_nonexistent_fails() {
        let mut b = ready_backend();
        assert!(b.free(9999).is_err());
    }

    #[test]
    fn backend_allocation_write_read() {
        let mut b = ready_backend();
        let id = b.allocate(64, MemoryKind::Device).unwrap();
        b.get_allocation_mut(id).unwrap().write(0, &[1, 2, 3]).unwrap();
        let data = b.get_allocation(id).unwrap().read(0, 3).unwrap();
        assert_eq!(data, &[1, 2, 3]);
    }

    #[test]
    fn backend_load_module() {
        let mut b = ready_backend();
        let mid = b.load_module("m", ModuleFormat::PTX, b".version 7.0", vec!["k".into()]).unwrap();
        let m = b.get_module(mid).unwrap();
        assert_eq!(m.name(), "m");
    }

    #[test]
    fn backend_unload_module() {
        let mut b = ready_backend();
        let mid = b.load_module("m", ModuleFormat::PTX, b"data", vec!["k".into()]).unwrap();
        b.unload_module(mid).unwrap();
        assert!(b.get_module(mid).is_err());
    }

    #[test]
    fn backend_unload_nonexistent_fails() {
        let mut b = ready_backend();
        assert!(b.unload_module(9999).is_err());
    }

    #[test]
    fn backend_stream_mut() {
        let mut b = ready_backend();
        let s = b.stream_mut(0).unwrap();
        s.record_work();
        assert_eq!(s.work_items(), 1);
    }

    #[test]
    fn backend_stream_mut_out_of_range() {
        let mut b = ready_backend();
        assert!(b.stream_mut(100).is_err());
    }

    #[test]
    fn backend_synchronize_all() {
        let mut b = ready_backend();
        b.stream_mut(0).unwrap().record_work();
        b.stream_mut(1).unwrap().record_work();
        b.synchronize_all().unwrap();
        assert!(b.streams().iter().all(CUDAStream::is_synchronized));
    }

    #[test]
    fn backend_peer_access() {
        let mut b = ready_backend();
        b.peer_access_mut().enable(0, 1).unwrap();
        assert!(b.peer_access().is_enabled(0, 1));
    }

    #[test]
    fn backend_shutdown() {
        let mut b = ready_backend();
        b.allocate(128, MemoryKind::Device).unwrap();
        b.load_module("m", ModuleFormat::PTX, b"x", vec!["k".into()]).unwrap();
        b.shutdown().unwrap();
        assert_eq!(b.state(), BackendState::Shutdown);
        assert_eq!(b.total_allocated_bytes(), 0);
    }

    #[test]
    fn backend_shutdown_double_fails() {
        let mut b = ready_backend();
        b.shutdown().unwrap();
        assert!(b.shutdown().is_err());
    }

    #[test]
    fn backend_operations_after_shutdown_fail() {
        let mut b = ready_backend();
        b.shutdown().unwrap();
        assert!(b.allocate(64, MemoryKind::Device).is_err());
        assert!(b.synchronize_all().is_err());
    }

    #[test]
    fn backend_operations_before_init_fail() {
        let mut b = CUDABackend::new(CUDAConfig::default());
        assert!(b.allocate(64, MemoryKind::Device).is_err());
    }

    #[test]
    fn backend_display() {
        let b = ready_backend();
        let s = b.to_string();
        assert!(s.contains("CUDABackend"));
        assert!(s.contains("ready"));
    }

    #[test]
    fn backend_state_display() {
        assert_eq!(BackendState::Uninitialised.to_string(), "uninitialised");
        assert_eq!(BackendState::Ready.to_string(), "ready");
        assert_eq!(BackendState::Shutdown.to_string(), "shutdown");
    }

    #[test]
    fn backend_config_accessor() {
        let b = ready_backend();
        assert_eq!(b.config().device_index, 0);
    }

    #[test]
    fn backend_select_second_device() {
        let cfg = CUDAConfig::for_device(1);
        let mut b = CUDABackend::new(cfg);
        b.init(&test_devices()).unwrap();
        assert_eq!(b.device().unwrap().index, 1);
        assert_eq!(b.device().unwrap().name, "Test GPU 1");
    }

    // ── Error display tests ──────────────────────────────────────────

    #[test]
    fn error_display_coverage() {
        let cases: Vec<CUDAError> = vec![
            CUDAError::InvalidDevice(7),
            CUDAError::InvalidStream(42),
            CUDAError::AllocationFailed("oom".into()),
            CUDAError::ModuleLoadFailed("bad ptx".into()),
            CUDAError::KernelNotFound("foo".into()),
            CUDAError::InvalidLaunchConfig("zero block".into()),
            CUDAError::GraphError("nope".into()),
            CUDAError::PeerAccessFailed { from: 0, to: 1 },
            CUDAError::Backend("misc".into()),
        ];
        for e in &cases {
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(CUDAError::Backend("test".into()));
        assert!(!e.to_string().is_empty());
    }

    // ── Integration-style tests ──────────────────────────────────────

    #[test]
    fn integration_full_lifecycle() {
        let mut backend =
            CUDABackend::new(CUDAConfig::default().with_stream_count(3).with_graph_capture(true));
        backend.init(&test_devices()).unwrap();

        // allocate
        let buf_id = backend.allocate(4096, MemoryKind::Device).unwrap();
        backend.get_allocation_mut(buf_id).unwrap().write(0, &[0xFF; 32]).unwrap();

        // load module + create kernel
        let mod_id = backend
            .load_module("ops", ModuleFormat::PTX, b".version 8.0\n", vec!["gemm".into()])
            .unwrap();
        let module = backend.get_module(mod_id).unwrap();
        let mut kernel = CUDAKernel::new(module, "gemm").unwrap();
        kernel.set_grid(Dim3::x_only(16));
        kernel.set_block(Dim3::x_only(256));
        kernel.set_shared_memory(8192);

        // launch on first stream
        let dev = backend.device().unwrap().clone();
        let stream = backend.stream_mut(0).unwrap();
        kernel.launch(stream, &dev).unwrap();
        assert_eq!(kernel.launch_count(), 1);

        // event timing
        let stream = backend.stream_mut(0).unwrap();
        let mut start = CUDAEvent::new("start");
        start.record(stream).unwrap();
        std::thread::sleep(Duration::from_millis(1));
        let mut end = CUDAEvent::new("end");
        let stream = backend.stream_mut(0).unwrap();
        end.record(stream).unwrap();
        let _dt = end.elapsed_since(&start).unwrap();

        // sync + shutdown
        backend.synchronize_all().unwrap();
        backend.free(buf_id).unwrap();
        backend.unload_module(mod_id).unwrap();
        backend.shutdown().unwrap();
    }

    #[test]
    fn integration_graph_capture_replay() {
        let mut backend = ready_backend();
        let dev = backend.device().unwrap().clone();
        let mod_id =
            backend.load_module("g", ModuleFormat::PTX, b"data", vec!["k".into()]).unwrap();
        let module = backend.get_module(mod_id).unwrap();
        let mut kernel = CUDAKernel::new(module, "k").unwrap();
        kernel.set_grid(Dim3::x_only(4));
        kernel.set_block(Dim3::x_only(64));

        let mut graph = CUDAGraph::new("inference");
        graph.begin_capture().unwrap();
        // simulate captured ops
        let stream = backend.stream_mut(0).unwrap();
        kernel.launch(stream, &dev).unwrap();
        graph.record_op().unwrap();
        graph.record_op().unwrap();
        graph.end_capture().unwrap();

        // replay 3 times
        for _ in 0..3 {
            let stream = backend.stream_mut(0).unwrap();
            graph.replay(stream).unwrap();
        }
        assert_eq!(graph.replay_count(), 3);

        backend.shutdown().unwrap();
    }

    #[test]
    fn integration_multi_device_peer_access() {
        let cfg = CUDAConfig::for_device(0);
        let mut b = CUDABackend::new(cfg);
        b.init(&test_devices()).unwrap();
        b.peer_access_mut().enable(0, 1).unwrap();
        b.peer_access_mut().enable(1, 0).unwrap();
        assert_eq!(b.peer_access().active_count(), 2);
        b.peer_access_mut().disable(0, 1);
        assert_eq!(b.peer_access().active_count(), 1);
        b.shutdown().unwrap();
    }
}
