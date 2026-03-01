//! Module stub - implementation pending merge from feature branch
//! Apple Metal backend for GPU compute on macOS/iOS.
//!
//! CPU reference implementation of a Metal-style GPU compute abstraction.
//! Provides device discovery, command queue / buffer lifecycle, shader library
//! compilation, compute pipeline dispatch, fencing, and GPU frame capture scopes.
//!
//! All operations are executed on the CPU so the module compiles and tests on
//! every platform. A real Metal FFI layer can replace the inner implementations
//! behind the same public API surface.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Identifiers ─────────────────────────────────────────────────────────────

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_FENCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_buffer_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

fn next_fence_id() -> u64 {
    NEXT_FENCE_ID.fetch_add(1, Ordering::Relaxed)
}

// ── MetalConfig ─────────────────────────────────────────────────────────────

/// Configuration for initialising a Metal device and command infrastructure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalConfig {
    /// Zero-based GPU device index.
    pub device_index: usize,
    /// Number of command buffers the queue may keep in flight.
    pub command_buffer_count: usize,
    /// Whether GPU frame capture is enabled for debugging.
    pub capture_enabled: bool,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self { device_index: 0, command_buffer_count: 3, capture_enabled: false }
    }
}

impl MetalConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the device index.
    #[must_use]
    pub const fn with_device_index(mut self, index: usize) -> Self {
        self.device_index = index;
        self
    }

    /// Set the command buffer count.
    #[must_use]
    pub const fn with_command_buffer_count(mut self, count: usize) -> Self {
        self.command_buffer_count = count;
        self
    }

    /// Enable or disable GPU frame capture.
    #[must_use]
    pub const fn with_capture_enabled(mut self, enabled: bool) -> Self {
        self.capture_enabled = enabled;
        self
    }
}

impl fmt::Display for MetalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalConfig(device={}, buffers={}, capture={})",
            self.device_index, self.command_buffer_count, self.capture_enabled,
        )
    }
}

// ── MetalDevice ─────────────────────────────────────────────────────────────

/// Properties reported by a Metal GPU device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProperties {
    /// Human-readable device name.
    pub name: String,
    /// Whether the device uses unified (shared) memory.
    pub unified_memory: bool,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: usize,
    /// Maximum buffer length in bytes.
    pub max_buffer_length: usize,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "CPU Reference Device".to_string(),
            unified_memory: true,
            max_threads_per_threadgroup: 1024,
            max_buffer_length: 256 * 1024 * 1024,
        }
    }
}

/// A Metal GPU device handle.
#[derive(Debug, Clone)]
pub struct MetalDevice {
    /// Zero-based index of this device.
    index: usize,
    /// Device properties.
    properties: DeviceProperties,
}

impl MetalDevice {
    /// Open a device by index.  The CPU reference always succeeds.
    #[must_use]
    pub fn open(index: usize) -> Self {
        Self { index, properties: DeviceProperties::default() }
    }

    /// Open a device with explicit properties.
    #[must_use]
    pub const fn with_properties(index: usize, properties: DeviceProperties) -> Self {
        Self { index, properties }
    }

    /// Device index.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Borrow the device properties.
    #[must_use]
    pub const fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Device name (convenience accessor).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.properties.name
    }

    /// Whether the device reports unified memory.
    #[must_use]
    pub const fn has_unified_memory(&self) -> bool {
        self.properties.unified_memory
    }

    /// Maximum threadgroup width.
    #[must_use]
    pub const fn max_threads_per_threadgroup(&self) -> usize {
        self.properties.max_threads_per_threadgroup
    }

    /// Validate that a requested buffer size fits within device limits.
    #[must_use]
    pub const fn supports_buffer_size(&self, bytes: usize) -> bool {
        bytes <= self.properties.max_buffer_length
    }
}

impl fmt::Display for MetalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalDevice[{}] \"{}\" unified={} max_tpg={}",
            self.index,
            self.properties.name,
            self.properties.unified_memory,
            self.properties.max_threads_per_threadgroup,
        )
    }
}

// ── MetalCommandQueue ───────────────────────────────────────────────────────

/// Status of a command buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferStatus {
    /// Buffer is available for encoding.
    Available,
    /// Buffer is currently being encoded.
    Encoding,
    /// Buffer has been committed and is executing.
    Committed,
    /// Buffer has completed execution.
    Completed,
}

impl fmt::Display for CommandBufferStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Available => write!(f, "Available"),
            Self::Encoding => write!(f, "Encoding"),
            Self::Committed => write!(f, "Committed"),
            Self::Completed => write!(f, "Completed"),
        }
    }
}

/// Manages a pool of command buffers and their lifecycle.
#[derive(Debug)]
pub struct MetalCommandQueue {
    /// Maximum number of in-flight buffers.
    capacity: usize,
    /// Current status of each slot.
    slots: Vec<CommandBufferStatus>,
    /// Total buffers committed since creation.
    total_committed: u64,
}

impl MetalCommandQueue {
    /// Create a new queue with the given number of buffer slots.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self { capacity: cap, slots: vec![CommandBufferStatus::Available; cap], total_committed: 0 }
    }

    /// Queue capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of slots currently available for encoding.
    #[must_use]
    pub fn available_slots(&self) -> usize {
        self.slots.iter().filter(|s| **s == CommandBufferStatus::Available).count()
    }

    /// Try to acquire a slot for encoding.  Returns `Some(slot_index)` on
    /// success.
    pub fn acquire(&mut self) -> Option<usize> {
        if let Some(idx) = self.slots.iter().position(|s| *s == CommandBufferStatus::Available) {
            self.slots[idx] = CommandBufferStatus::Encoding;
            Some(idx)
        } else {
            None
        }
    }

    /// Commit a slot (move from Encoding → Committed).
    ///
    /// Returns `true` if the transition succeeded.
    pub fn commit(&mut self, slot: usize) -> bool {
        if slot < self.capacity && self.slots[slot] == CommandBufferStatus::Encoding {
            self.slots[slot] = CommandBufferStatus::Committed;
            self.total_committed += 1;
            true
        } else {
            false
        }
    }

    /// Complete a slot (move from Committed → Completed).
    pub fn complete(&mut self, slot: usize) -> bool {
        if slot < self.capacity && self.slots[slot] == CommandBufferStatus::Committed {
            self.slots[slot] = CommandBufferStatus::Completed;
            true
        } else {
            false
        }
    }

    /// Recycle a completed slot back to Available.
    pub fn recycle(&mut self, slot: usize) -> bool {
        if slot < self.capacity && self.slots[slot] == CommandBufferStatus::Completed {
            self.slots[slot] = CommandBufferStatus::Available;
            true
        } else {
            false
        }
    }

    /// Reset all slots to Available.
    pub fn reset(&mut self) {
        for s in &mut self.slots {
            *s = CommandBufferStatus::Available;
        }
    }

    /// Status of a specific slot.
    #[must_use]
    pub fn slot_status(&self, slot: usize) -> Option<CommandBufferStatus> {
        self.slots.get(slot).copied()
    }

    /// Total buffers committed over the lifetime of this queue.
    #[must_use]
    pub const fn total_committed(&self) -> u64 {
        self.total_committed
    }
}

// ── MetalBuffer ─────────────────────────────────────────────────────────────

/// Storage mode for a Metal buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    /// Shared between CPU and GPU (unified memory).
    Shared,
    /// Managed: explicit synchronisation required.
    Managed,
    /// GPU-private; CPU cannot access contents.
    Private,
}

impl fmt::Display for StorageMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared => write!(f, "Shared"),
            Self::Managed => write!(f, "Managed"),
            Self::Private => write!(f, "Private"),
        }
    }
}

/// A typed GPU buffer backed by a CPU `Vec<T>`.
#[derive(Debug, Clone)]
pub struct MetalBuffer<T: Clone + fmt::Debug> {
    id: u64,
    label: String,
    storage_mode: StorageMode,
    data: Vec<T>,
}

impl<T: Clone + fmt::Debug> MetalBuffer<T> {
    /// Allocate a buffer with the given length, filled by cloning `init`.
    #[must_use]
    pub fn new(label: &str, len: usize, init: T, mode: StorageMode) -> Self {
        Self {
            id: next_buffer_id(),
            label: label.to_string(),
            storage_mode: mode,
            data: vec![init; len],
        }
    }

    /// Create a buffer from existing data.
    #[must_use]
    pub fn from_data(label: &str, data: Vec<T>, mode: StorageMode) -> Self {
        Self { id: next_buffer_id(), label: label.to_string(), storage_mode: mode, data }
    }

    /// Unique buffer identifier.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Buffer label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Storage mode.
    #[must_use]
    pub const fn storage_mode(&self) -> StorageMode {
        self.storage_mode
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Size in bytes (element count × size of `T`).
    #[must_use]
    pub const fn byte_length(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }

    /// Immutable access to the contents.
    ///
    /// # Panics
    ///
    /// Panics when the buffer storage mode is `Private`.
    #[must_use]
    pub fn contents(&self) -> &[T] {
        assert!(self.storage_mode != StorageMode::Private, "cannot read Private buffer on CPU");
        &self.data
    }

    /// Mutable access to the contents.
    ///
    /// # Panics
    ///
    /// Panics when the buffer storage mode is `Private`.
    pub fn contents_mut(&mut self) -> &mut [T] {
        assert!(self.storage_mode != StorageMode::Private, "cannot write Private buffer on CPU");
        &mut self.data
    }
}

impl<T: Clone + fmt::Debug> fmt::Display for MetalBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalBuffer[{}] \"{}\" len={} mode={}",
            self.id,
            self.label,
            self.data.len(),
            self.storage_mode,
        )
    }
}

// ── MetalLibrary ────────────────────────────────────────────────────────────

/// Source from which a shader library was compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibrarySource {
    /// Compiled from Metal Shading Language source at runtime.
    Source(String),
    /// Loaded from a pre-compiled `.metallib` binary.
    Precompiled(String),
}

/// A compiled Metal shader library exposing named kernel functions.
#[derive(Debug, Clone)]
pub struct MetalLibrary {
    label: String,
    source: LibrarySource,
    functions: Vec<String>,
}

impl MetalLibrary {
    /// Compile a library from MSL source.
    ///
    /// The CPU reference implementation parses function names from `kernel
    /// void name(` patterns.
    #[must_use]
    pub fn from_source(label: &str, source: &str) -> Self {
        let functions = parse_kernel_names(source);
        Self {
            label: label.to_string(),
            source: LibrarySource::Source(source.to_string()),
            functions,
        }
    }

    /// Load a pre-compiled library with explicit function names.
    #[must_use]
    pub fn from_precompiled(label: &str, path: &str, functions: Vec<String>) -> Self {
        Self {
            label: label.to_string(),
            source: LibrarySource::Precompiled(path.to_string()),
            functions,
        }
    }

    /// Library label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Library source.
    #[must_use]
    pub const fn source(&self) -> &LibrarySource {
        &self.source
    }

    /// List of kernel function names exposed by this library.
    #[must_use]
    pub fn function_names(&self) -> &[String] {
        &self.functions
    }

    /// Look up a function by name.
    #[must_use]
    pub fn get_function(&self, name: &str) -> Option<&str> {
        self.functions.iter().find(|f| f == &name).map(String::as_str)
    }
}

/// Extracts function names from simplified MSL-like kernel declarations.
fn parse_kernel_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("kernel void ")
            && let Some(paren) = rest.find('(')
        {
            let name = rest[..paren].trim();
            if !name.is_empty() {
                names.push(name.to_string());
            }
        }
    }
    names
}

impl fmt::Display for MetalLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalLibrary \"{}\" functions=[{}]", self.label, self.functions.join(", "),)
    }
}

// ── MetalComputePipeline ────────────────────────────────────────────────────

/// A compute pipeline state bound to a specific kernel function.
#[derive(Debug, Clone)]
pub struct MetalComputePipeline {
    function_name: String,
    max_total_threads: usize,
    thread_execution_width: usize,
}

impl MetalComputePipeline {
    /// Create a pipeline for the named kernel function.
    #[must_use]
    pub fn new(function_name: &str) -> Self {
        Self {
            function_name: function_name.to_string(),
            max_total_threads: 1024,
            thread_execution_width: 32,
        }
    }

    /// Create with explicit thread dimensions.
    #[must_use]
    pub fn with_threads(function_name: &str, max_total: usize, execution_width: usize) -> Self {
        Self {
            function_name: function_name.to_string(),
            max_total_threads: max_total,
            thread_execution_width: execution_width,
        }
    }

    /// Kernel function name.
    #[must_use]
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Maximum total threads per threadgroup.
    #[must_use]
    pub const fn max_total_threads(&self) -> usize {
        self.max_total_threads
    }

    /// SIMD execution width.
    #[must_use]
    pub const fn thread_execution_width(&self) -> usize {
        self.thread_execution_width
    }
}

impl fmt::Display for MetalComputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalComputePipeline \"{}\" max_threads={} width={}",
            self.function_name, self.max_total_threads, self.thread_execution_width,
        )
    }
}

// ── MetalCommandEncoder ─────────────────────────────────────────────────────

/// A dispatch dimensions descriptor (threadgroups × threads-per-threadgroup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchSize {
    /// Threadgroups along X.
    pub threadgroups_x: usize,
    /// Threadgroups along Y.
    pub threadgroups_y: usize,
    /// Threadgroups along Z.
    pub threadgroups_z: usize,
    /// Threads per threadgroup along X.
    pub threads_per_tg_x: usize,
    /// Threads per threadgroup along Y.
    pub threads_per_tg_y: usize,
    /// Threads per threadgroup along Z.
    pub threads_per_tg_z: usize,
}

impl DispatchSize {
    /// Simple 1-D dispatch.
    #[must_use]
    pub const fn one_d(threadgroups: usize, threads_per_tg: usize) -> Self {
        Self {
            threadgroups_x: threadgroups,
            threadgroups_y: 1,
            threadgroups_z: 1,
            threads_per_tg_x: threads_per_tg,
            threads_per_tg_y: 1,
            threads_per_tg_z: 1,
        }
    }

    /// 2-D dispatch.
    #[must_use]
    pub const fn two_d(tg_x: usize, tg_y: usize, tpt_x: usize, tpt_y: usize) -> Self {
        Self {
            threadgroups_x: tg_x,
            threadgroups_y: tg_y,
            threadgroups_z: 1,
            threads_per_tg_x: tpt_x,
            threads_per_tg_y: tpt_y,
            threads_per_tg_z: 1,
        }
    }

    /// Total threads dispatched.
    #[must_use]
    pub const fn total_threads(&self) -> usize {
        self.threadgroups_x
            * self.threadgroups_y
            * self.threadgroups_z
            * self.threads_per_tg_x
            * self.threads_per_tg_y
            * self.threads_per_tg_z
    }
}

/// A recorded buffer binding in the encoder.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    /// Buffer identifier.
    pub buffer_id: u64,
    /// Binding index.
    pub index: usize,
    /// Byte offset within the buffer.
    pub offset: usize,
}

/// Compute command encoder that records bindings and dispatch calls.
#[derive(Debug)]
pub struct MetalCommandEncoder {
    pipeline_name: String,
    bindings: Vec<BufferBinding>,
    dispatches: Vec<DispatchSize>,
    ended: bool,
}

impl MetalCommandEncoder {
    /// Begin encoding for the given pipeline.
    #[must_use]
    pub fn new(pipeline: &MetalComputePipeline) -> Self {
        Self {
            pipeline_name: pipeline.function_name().to_string(),
            bindings: Vec::new(),
            dispatches: Vec::new(),
            ended: false,
        }
    }

    /// Bind a buffer at the given index with an optional byte offset.
    ///
    /// # Panics
    ///
    /// Panics if the encoder has already been ended.
    pub fn set_buffer<T: Clone + fmt::Debug>(
        &mut self,
        buffer: &MetalBuffer<T>,
        index: usize,
        offset: usize,
    ) {
        assert!(!self.ended, "encoder already ended");
        self.bindings.push(BufferBinding { buffer_id: buffer.id(), index, offset });
    }

    /// Record a dispatch call.
    ///
    /// # Panics
    ///
    /// Panics if the encoder has already been ended.
    pub fn dispatch(&mut self, size: DispatchSize) {
        assert!(!self.ended, "encoder already ended");
        self.dispatches.push(size);
    }

    /// End encoding.  No further bindings or dispatches may be recorded.
    pub const fn end(&mut self) {
        self.ended = true;
    }

    /// Whether encoding has ended.
    #[must_use]
    pub const fn is_ended(&self) -> bool {
        self.ended
    }

    /// Pipeline name this encoder targets.
    #[must_use]
    pub fn pipeline_name(&self) -> &str {
        &self.pipeline_name
    }

    /// Recorded buffer bindings.
    #[must_use]
    pub fn bindings(&self) -> &[BufferBinding] {
        &self.bindings
    }

    /// Recorded dispatch calls.
    #[must_use]
    pub fn dispatches(&self) -> &[DispatchSize] {
        &self.dispatches
    }
}

// ── MetalFence ──────────────────────────────────────────────────────────────

/// GPU-CPU synchronisation primitive.
#[derive(Debug)]
pub struct MetalFence {
    id: u64,
    label: String,
    signalled: bool,
}

impl MetalFence {
    /// Create a new unsignalled fence.
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self { id: next_fence_id(), label: label.to_string(), signalled: false }
    }

    /// Fence identifier.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Fence label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Whether the fence has been signalled.
    #[must_use]
    pub const fn is_signalled(&self) -> bool {
        self.signalled
    }

    /// Signal the fence (GPU side completes).
    pub const fn signal(&mut self) {
        self.signalled = true;
    }

    /// Wait for the fence.  In the CPU reference this is a no-op if already
    /// signalled, otherwise it immediately signals.
    pub const fn wait(&mut self) {
        self.signalled = true;
    }

    /// Reset to unsignalled state.
    pub const fn reset(&mut self) {
        self.signalled = false;
    }
}

impl fmt::Display for MetalFence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetalFence[{}] \"{}\" signalled={}", self.id, self.label, self.signalled,)
    }
}

// ── MetalCaptureScope ───────────────────────────────────────────────────────

/// GPU frame capture scope for debugging tools (e.g. Xcode GPU Debugger).
#[derive(Debug)]
pub struct MetalCaptureScope {
    label: String,
    active: bool,
    frame_count: u64,
}

impl MetalCaptureScope {
    /// Create a new capture scope.
    #[must_use]
    pub fn new(label: &str) -> Self {
        Self { label: label.to_string(), active: false, frame_count: 0 }
    }

    /// Scope label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Whether capture is currently active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Number of frames captured so far.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Begin a capture frame.
    ///
    /// # Panics
    ///
    /// Panics if a frame is already active.
    pub fn begin(&mut self) {
        assert!(!self.active, "capture frame already active");
        self.active = true;
    }

    /// End the current capture frame.
    ///
    /// # Panics
    ///
    /// Panics if no frame is active.
    pub fn end(&mut self) {
        assert!(self.active, "no capture frame active");
        self.active = false;
        self.frame_count += 1;
    }
}

impl fmt::Display for MetalCaptureScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalCaptureScope \"{}\" active={} frames={}",
            self.label, self.active, self.frame_count,
        )
    }
}

// ── MetalBackend ────────────────────────────────────────────────────────────

/// Result of a single backend dispatch operation.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Kernel function name.
    pub function_name: String,
    /// Total threads dispatched.
    pub total_threads: usize,
    /// Wall-clock duration of the (simulated) dispatch.
    pub duration: Duration,
    /// Whether the dispatch completed successfully.
    pub success: bool,
    /// Human-readable status message.
    pub message: String,
}

/// Orchestrator that drives the full Metal pipeline:
/// device → queue → library → pipeline → encode → dispatch.
#[derive(Debug)]
pub struct MetalBackend {
    config: MetalConfig,
    device: MetalDevice,
    queue: MetalCommandQueue,
    libraries: HashMap<String, MetalLibrary>,
    pipelines: HashMap<String, MetalComputePipeline>,
    capture_scope: Option<MetalCaptureScope>,
    dispatch_count: u64,
}

impl MetalBackend {
    /// Initialise the backend from a configuration.
    #[must_use]
    pub fn new(config: MetalConfig) -> Self {
        let device = MetalDevice::open(config.device_index);
        let queue = MetalCommandQueue::new(config.command_buffer_count);
        let capture_scope =
            if config.capture_enabled { Some(MetalCaptureScope::new("default")) } else { None };

        Self {
            config,
            device,
            queue,
            libraries: HashMap::new(),
            pipelines: HashMap::new(),
            capture_scope,
            dispatch_count: 0,
        }
    }

    /// Borrow the configuration.
    #[must_use]
    pub const fn config(&self) -> &MetalConfig {
        &self.config
    }

    /// Borrow the device.
    #[must_use]
    pub const fn device(&self) -> &MetalDevice {
        &self.device
    }

    /// Mutable access to the command queue.
    pub const fn queue_mut(&mut self) -> &mut MetalCommandQueue {
        &mut self.queue
    }

    /// Borrow the command queue.
    #[must_use]
    pub const fn queue(&self) -> &MetalCommandQueue {
        &self.queue
    }

    /// Number of dispatches executed.
    #[must_use]
    pub const fn dispatch_count(&self) -> u64 {
        self.dispatch_count
    }

    /// Register a shader library compiled from source.
    pub fn add_library_from_source(&mut self, label: &str, source: &str) {
        let lib = MetalLibrary::from_source(label, source);
        self.libraries.insert(label.to_string(), lib);
    }

    /// Register a pre-compiled shader library.
    pub fn add_precompiled_library(&mut self, label: &str, path: &str, functions: Vec<String>) {
        let lib = MetalLibrary::from_precompiled(label, path, functions);
        self.libraries.insert(label.to_string(), lib);
    }

    /// Borrow a registered library by label.
    #[must_use]
    pub fn get_library(&self, label: &str) -> Option<&MetalLibrary> {
        self.libraries.get(label)
    }

    /// Create a compute pipeline for a kernel function found in any
    /// registered library.
    pub fn create_pipeline(&mut self, function_name: &str) -> Option<&MetalComputePipeline> {
        let found = self.libraries.values().any(|lib| lib.get_function(function_name).is_some());
        if !found {
            return None;
        }
        let pipeline = MetalComputePipeline::new(function_name);
        self.pipelines.insert(function_name.to_string(), pipeline);
        self.pipelines.get(function_name)
    }

    /// Borrow a previously created pipeline.
    #[must_use]
    pub fn get_pipeline(&self, function_name: &str) -> Option<&MetalComputePipeline> {
        self.pipelines.get(function_name)
    }

    /// Execute a full dispatch: acquire slot → encode → dispatch → commit →
    /// complete → recycle.
    ///
    /// Returns `None` if no command buffer slot is available.
    pub fn dispatch(&mut self, function_name: &str, size: DispatchSize) -> Option<DispatchResult> {
        let pipeline = self.pipelines.get(function_name)?;
        let slot = self.queue.acquire()?;

        let start = Instant::now();

        // Encode
        let mut encoder = MetalCommandEncoder::new(pipeline);
        encoder.dispatch(size);
        encoder.end();

        // Commit → complete → recycle
        self.queue.commit(slot);
        self.queue.complete(slot);
        self.queue.recycle(slot);

        let elapsed = start.elapsed();
        self.dispatch_count += 1;

        Some(DispatchResult {
            function_name: function_name.to_string(),
            total_threads: size.total_threads(),
            duration: elapsed,
            success: true,
            message: format!("dispatched {} threads in {elapsed:?}", size.total_threads()),
        })
    }

    /// Borrow the capture scope (if capture is enabled).
    #[must_use]
    pub const fn capture_scope(&self) -> Option<&MetalCaptureScope> {
        self.capture_scope.as_ref()
    }

    /// Mutable access to the capture scope.
    pub const fn capture_scope_mut(&mut self) -> Option<&mut MetalCaptureScope> {
        self.capture_scope.as_mut()
    }

    /// Reset the backend: clear pipelines, reset queue, zero dispatch count.
    pub fn reset(&mut self) {
        self.pipelines.clear();
        self.queue.reset();
        self.dispatch_count = 0;
    }
}

impl fmt::Display for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MetalBackend device=\"{}\" libs={} pipelines={} dispatches={}",
            self.device.name(),
            self.libraries.len(),
            self.pipelines.len(),
            self.dispatch_count,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // ── helpers ─────────────────────────────────────────────────────────

    const SAMPLE_MSL: &str = "\
kernel void add_arrays(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

kernel void scale_array(device float* data [[buffer(0)]],
                        constant float& factor [[buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
    data[index] *= factor;
}
";

    fn sample_config() -> MetalConfig {
        MetalConfig::new()
    }

    fn capture_config() -> MetalConfig {
        MetalConfig::new().with_capture_enabled(true)
    }

    // ── MetalConfig ─────────────────────────────────────────────────────

    #[test]
    fn config_default_device_index() {
        assert_eq!(MetalConfig::default().device_index, 0);
    }

    #[test]
    fn config_default_buffer_count() {
        assert_eq!(MetalConfig::default().command_buffer_count, 3);
    }

    #[test]
    fn config_default_capture_disabled() {
        assert!(!MetalConfig::default().capture_enabled);
    }

    #[test]
    fn config_builder_device_index() {
        let c = MetalConfig::new().with_device_index(2);
        assert_eq!(c.device_index, 2);
    }

    #[test]
    fn config_builder_buffer_count() {
        let c = MetalConfig::new().with_command_buffer_count(5);
        assert_eq!(c.command_buffer_count, 5);
    }

    #[test]
    fn config_builder_capture() {
        let c = MetalConfig::new().with_capture_enabled(true);
        assert!(c.capture_enabled);
    }

    #[test]
    fn config_builder_chained() {
        let c = MetalConfig::new()
            .with_device_index(1)
            .with_command_buffer_count(8)
            .with_capture_enabled(true);
        assert_eq!(c.device_index, 1);
        assert_eq!(c.command_buffer_count, 8);
        assert!(c.capture_enabled);
    }

    #[test]
    fn config_display() {
        let s = format!("{}", MetalConfig::default());
        assert!(s.contains("MetalConfig"));
        assert!(s.contains("device=0"));
    }

    #[test]
    fn config_eq() {
        assert_eq!(MetalConfig::default(), MetalConfig::new());
    }

    // ── MetalDevice ─────────────────────────────────────────────────────

    #[test]
    fn device_open_index() {
        let d = MetalDevice::open(0);
        assert_eq!(d.index(), 0);
    }

    #[test]
    fn device_default_name() {
        let d = MetalDevice::open(0);
        assert_eq!(d.name(), "CPU Reference Device");
    }

    #[test]
    fn device_unified_memory() {
        let d = MetalDevice::open(0);
        assert!(d.has_unified_memory());
    }

    #[test]
    fn device_max_threads() {
        let d = MetalDevice::open(0);
        assert_eq!(d.max_threads_per_threadgroup(), 1024);
    }

    #[test]
    fn device_supports_small_buffer() {
        let d = MetalDevice::open(0);
        assert!(d.supports_buffer_size(1024));
    }

    #[test]
    fn device_rejects_huge_buffer() {
        let d = MetalDevice::open(0);
        assert!(!d.supports_buffer_size(usize::MAX));
    }

    #[test]
    fn device_custom_properties() {
        let props = DeviceProperties {
            name: "Apple M2".to_string(),
            unified_memory: true,
            max_threads_per_threadgroup: 512,
            max_buffer_length: 128 * 1024 * 1024,
        };
        let d = MetalDevice::with_properties(1, props);
        assert_eq!(d.index(), 1);
        assert_eq!(d.name(), "Apple M2");
        assert_eq!(d.max_threads_per_threadgroup(), 512);
    }

    #[test]
    fn device_display() {
        let s = format!("{}", MetalDevice::open(0));
        assert!(s.contains("MetalDevice[0]"));
        assert!(s.contains("CPU Reference Device"));
    }

    #[test]
    fn device_properties_borrow() {
        let d = MetalDevice::open(0);
        let p = d.properties();
        assert_eq!(p.name, "CPU Reference Device");
        assert!(p.unified_memory);
    }

    // ── MetalCommandQueue ───────────────────────────────────────────────

    #[test]
    fn queue_capacity() {
        let q = MetalCommandQueue::new(4);
        assert_eq!(q.capacity(), 4);
    }

    #[test]
    fn queue_minimum_capacity() {
        let q = MetalCommandQueue::new(0);
        assert_eq!(q.capacity(), 1);
    }

    #[test]
    fn queue_all_initially_available() {
        let q = MetalCommandQueue::new(3);
        assert_eq!(q.available_slots(), 3);
    }

    #[test]
    fn queue_acquire_reduces_available() {
        let mut q = MetalCommandQueue::new(3);
        q.acquire();
        assert_eq!(q.available_slots(), 2);
    }

    #[test]
    fn queue_acquire_returns_index() {
        let mut q = MetalCommandQueue::new(3);
        assert_eq!(q.acquire(), Some(0));
        assert_eq!(q.acquire(), Some(1));
        assert_eq!(q.acquire(), Some(2));
    }

    #[test]
    fn queue_acquire_exhaustion() {
        let mut q = MetalCommandQueue::new(1);
        assert!(q.acquire().is_some());
        assert!(q.acquire().is_none());
    }

    #[test]
    fn queue_commit_transitions() {
        let mut q = MetalCommandQueue::new(2);
        let slot = q.acquire().unwrap();
        assert!(q.commit(slot));
        assert_eq!(q.slot_status(slot), Some(CommandBufferStatus::Committed));
    }

    #[test]
    fn queue_commit_wrong_state_fails() {
        let mut q = MetalCommandQueue::new(2);
        assert!(!q.commit(0)); // still Available
    }

    #[test]
    fn queue_complete_after_commit() {
        let mut q = MetalCommandQueue::new(2);
        let slot = q.acquire().unwrap();
        q.commit(slot);
        assert!(q.complete(slot));
        assert_eq!(q.slot_status(slot), Some(CommandBufferStatus::Completed));
    }

    #[test]
    fn queue_recycle_after_complete() {
        let mut q = MetalCommandQueue::new(2);
        let slot = q.acquire().unwrap();
        q.commit(slot);
        q.complete(slot);
        assert!(q.recycle(slot));
        assert_eq!(q.slot_status(slot), Some(CommandBufferStatus::Available));
    }

    #[test]
    fn queue_recycle_wrong_state_fails() {
        let mut q = MetalCommandQueue::new(2);
        let slot = q.acquire().unwrap();
        assert!(!q.recycle(slot)); // still Encoding
    }

    #[test]
    fn queue_full_lifecycle() {
        let mut q = MetalCommandQueue::new(1);
        let s = q.acquire().unwrap();
        assert!(q.commit(s));
        assert!(q.complete(s));
        assert!(q.recycle(s));
        assert!(q.acquire().is_some());
    }

    #[test]
    fn queue_total_committed() {
        let mut q = MetalCommandQueue::new(2);
        let s0 = q.acquire().unwrap();
        let s1 = q.acquire().unwrap();
        q.commit(s0);
        q.commit(s1);
        assert_eq!(q.total_committed(), 2);
    }

    #[test]
    fn queue_reset_clears_all() {
        let mut q = MetalCommandQueue::new(3);
        q.acquire();
        q.acquire();
        q.reset();
        assert_eq!(q.available_slots(), 3);
    }

    #[test]
    fn queue_slot_status_out_of_range() {
        let q = MetalCommandQueue::new(2);
        assert_eq!(q.slot_status(99), None);
    }

    #[test]
    fn queue_commit_out_of_range() {
        let mut q = MetalCommandQueue::new(2);
        assert!(!q.commit(99));
    }

    #[test]
    fn command_buffer_status_display() {
        assert_eq!(format!("{}", CommandBufferStatus::Available), "Available");
        assert_eq!(format!("{}", CommandBufferStatus::Encoding), "Encoding");
        assert_eq!(format!("{}", CommandBufferStatus::Committed), "Committed");
        assert_eq!(format!("{}", CommandBufferStatus::Completed), "Completed");
    }

    // ── MetalBuffer ─────────────────────────────────────────────────────

    #[test]
    fn buffer_new_length() {
        let b = MetalBuffer::new("a", 128, 0.0_f32, StorageMode::Shared);
        assert_eq!(b.len(), 128);
    }

    #[test]
    fn buffer_new_init_value() {
        let b = MetalBuffer::new("a", 4, 42_u32, StorageMode::Shared);
        assert_eq!(b.contents(), &[42, 42, 42, 42]);
    }

    #[test]
    fn buffer_from_data() {
        let b = MetalBuffer::from_data("v", vec![1.0_f32, 2.0, 3.0], StorageMode::Shared);
        assert_eq!(b.len(), 3);
        assert_eq!(b.contents(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn buffer_label() {
        let b = MetalBuffer::new("weights", 1, 0_u8, StorageMode::Shared);
        assert_eq!(b.label(), "weights");
    }

    #[test]
    fn buffer_storage_mode() {
        let b = MetalBuffer::new("x", 1, 0_u8, StorageMode::Managed);
        assert_eq!(b.storage_mode(), StorageMode::Managed);
    }

    #[test]
    fn buffer_is_empty() {
        let b = MetalBuffer::from_data("e", Vec::<u8>::new(), StorageMode::Shared);
        assert!(b.is_empty());
    }

    #[test]
    fn buffer_byte_length() {
        let b = MetalBuffer::new("f", 10, 0.0_f32, StorageMode::Shared);
        assert_eq!(b.byte_length(), 10 * std::mem::size_of::<f32>());
    }

    #[test]
    fn buffer_contents_mut() {
        let mut b = MetalBuffer::new("m", 3, 0_i32, StorageMode::Shared);
        b.contents_mut()[1] = 99;
        assert_eq!(b.contents(), &[0, 99, 0]);
    }

    #[test]
    #[should_panic(expected = "cannot read Private buffer")]
    fn buffer_private_read_panics() {
        let b = MetalBuffer::new("p", 4, 0_u8, StorageMode::Private);
        let _ = b.contents();
    }

    #[test]
    #[should_panic(expected = "cannot write Private buffer")]
    fn buffer_private_write_panics() {
        let mut b = MetalBuffer::new("p", 4, 0_u8, StorageMode::Private);
        let _ = b.contents_mut();
    }

    #[test]
    fn buffer_unique_ids() {
        let a = MetalBuffer::new("a", 1, 0_u8, StorageMode::Shared);
        let b = MetalBuffer::new("b", 1, 0_u8, StorageMode::Shared);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn buffer_display() {
        let b = MetalBuffer::new("test", 16, 0_u8, StorageMode::Shared);
        let s = format!("{b}");
        assert!(s.contains("MetalBuffer"));
        assert!(s.contains("\"test\""));
        assert!(s.contains("len=16"));
    }

    #[test]
    fn storage_mode_display() {
        assert_eq!(format!("{}", StorageMode::Shared), "Shared");
        assert_eq!(format!("{}", StorageMode::Managed), "Managed");
        assert_eq!(format!("{}", StorageMode::Private), "Private");
    }

    // ── MetalLibrary ────────────────────────────────────────────────────

    #[test]
    fn library_from_source_parses_names() {
        let lib = MetalLibrary::from_source("test", SAMPLE_MSL);
        let names = lib.function_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"add_arrays".to_string()));
        assert!(names.contains(&"scale_array".to_string()));
    }

    #[test]
    fn library_label() {
        let lib = MetalLibrary::from_source("my_lib", "");
        assert_eq!(lib.label(), "my_lib");
    }

    #[test]
    fn library_source_variant() {
        let lib = MetalLibrary::from_source("s", "kernel void f(){}");
        assert!(matches!(lib.source(), LibrarySource::Source(_)));
    }

    #[test]
    fn library_precompiled_variant() {
        let lib = MetalLibrary::from_precompiled("pre", "lib.metallib", vec!["kern".to_string()]);
        assert!(matches!(lib.source(), LibrarySource::Precompiled(_)));
        assert_eq!(lib.function_names(), &["kern".to_string()]);
    }

    #[test]
    fn library_get_function_found() {
        let lib = MetalLibrary::from_source("test", SAMPLE_MSL);
        assert_eq!(lib.get_function("add_arrays"), Some("add_arrays"));
    }

    #[test]
    fn library_get_function_missing() {
        let lib = MetalLibrary::from_source("test", SAMPLE_MSL);
        assert_eq!(lib.get_function("no_such"), None);
    }

    #[test]
    fn library_empty_source() {
        let lib = MetalLibrary::from_source("empty", "// nothing here");
        assert!(lib.function_names().is_empty());
    }

    #[test]
    fn library_display() {
        let lib = MetalLibrary::from_source("test", SAMPLE_MSL);
        let s = format!("{lib}");
        assert!(s.contains("add_arrays"));
        assert!(s.contains("scale_array"));
    }

    #[test]
    fn parse_kernel_names_ignores_non_kernel() {
        let src = "void helper() {}\nkernel void real_kern(uint i) {}";
        let names = parse_kernel_names(src);
        assert_eq!(names, vec!["real_kern".to_string()]);
    }

    // ── MetalComputePipeline ────────────────────────────────────────────

    #[test]
    fn pipeline_function_name() {
        let p = MetalComputePipeline::new("add_arrays");
        assert_eq!(p.function_name(), "add_arrays");
    }

    #[test]
    fn pipeline_default_max_threads() {
        let p = MetalComputePipeline::new("f");
        assert_eq!(p.max_total_threads(), 1024);
    }

    #[test]
    fn pipeline_default_execution_width() {
        let p = MetalComputePipeline::new("f");
        assert_eq!(p.thread_execution_width(), 32);
    }

    #[test]
    fn pipeline_custom_threads() {
        let p = MetalComputePipeline::with_threads("f", 512, 64);
        assert_eq!(p.max_total_threads(), 512);
        assert_eq!(p.thread_execution_width(), 64);
    }

    #[test]
    fn pipeline_display() {
        let s = format!("{}", MetalComputePipeline::new("kern"));
        assert!(s.contains("kern"));
        assert!(s.contains("max_threads=1024"));
    }

    // ── DispatchSize ────────────────────────────────────────────────────

    #[test]
    fn dispatch_1d_total() {
        let d = DispatchSize::one_d(4, 256);
        assert_eq!(d.total_threads(), 1024);
    }

    #[test]
    fn dispatch_2d_total() {
        let d = DispatchSize::two_d(2, 3, 8, 8);
        assert_eq!(d.total_threads(), 2 * 3 * 8 * 8);
    }

    #[test]
    fn dispatch_1d_defaults_yz() {
        let d = DispatchSize::one_d(1, 1);
        assert_eq!(d.threadgroups_y, 1);
        assert_eq!(d.threadgroups_z, 1);
        assert_eq!(d.threads_per_tg_y, 1);
        assert_eq!(d.threads_per_tg_z, 1);
    }

    #[test]
    fn dispatch_size_eq() {
        let a = DispatchSize::one_d(4, 256);
        let b = DispatchSize::one_d(4, 256);
        assert_eq!(a, b);
    }

    // ── MetalCommandEncoder ─────────────────────────────────────────────

    #[test]
    fn encoder_pipeline_name() {
        let p = MetalComputePipeline::new("kern");
        let e = MetalCommandEncoder::new(&p);
        assert_eq!(e.pipeline_name(), "kern");
    }

    #[test]
    fn encoder_set_buffer_records_binding() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        let buf = MetalBuffer::new("a", 16, 0_u8, StorageMode::Shared);
        e.set_buffer(&buf, 0, 0);
        assert_eq!(e.bindings().len(), 1);
        assert_eq!(e.bindings()[0].index, 0);
    }

    #[test]
    fn encoder_multiple_bindings() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        let a = MetalBuffer::new("a", 4, 0_u8, StorageMode::Shared);
        let b = MetalBuffer::new("b", 4, 0_u8, StorageMode::Shared);
        e.set_buffer(&a, 0, 0);
        e.set_buffer(&b, 1, 0);
        assert_eq!(e.bindings().len(), 2);
    }

    #[test]
    fn encoder_dispatch_records() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        e.dispatch(DispatchSize::one_d(1, 256));
        assert_eq!(e.dispatches().len(), 1);
    }

    #[test]
    fn encoder_end_sets_flag() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        assert!(!e.is_ended());
        e.end();
        assert!(e.is_ended());
    }

    #[test]
    #[should_panic(expected = "encoder already ended")]
    fn encoder_set_buffer_after_end_panics() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        e.end();
        let buf = MetalBuffer::new("a", 1, 0_u8, StorageMode::Shared);
        e.set_buffer(&buf, 0, 0);
    }

    #[test]
    #[should_panic(expected = "encoder already ended")]
    fn encoder_dispatch_after_end_panics() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        e.end();
        e.dispatch(DispatchSize::one_d(1, 1));
    }

    #[test]
    fn encoder_binding_offset() {
        let p = MetalComputePipeline::new("kern");
        let mut e = MetalCommandEncoder::new(&p);
        let buf = MetalBuffer::new("a", 16, 0_u8, StorageMode::Shared);
        e.set_buffer(&buf, 0, 64);
        assert_eq!(e.bindings()[0].offset, 64);
    }

    // ── MetalFence ──────────────────────────────────────────────────────

    #[test]
    fn fence_initially_unsignalled() {
        let f = MetalFence::new("sync");
        assert!(!f.is_signalled());
    }

    #[test]
    fn fence_signal() {
        let mut f = MetalFence::new("sync");
        f.signal();
        assert!(f.is_signalled());
    }

    #[test]
    fn fence_wait_signals() {
        let mut f = MetalFence::new("sync");
        f.wait();
        assert!(f.is_signalled());
    }

    #[test]
    fn fence_reset() {
        let mut f = MetalFence::new("sync");
        f.signal();
        f.reset();
        assert!(!f.is_signalled());
    }

    #[test]
    fn fence_label() {
        let f = MetalFence::new("my_fence");
        assert_eq!(f.label(), "my_fence");
    }

    #[test]
    fn fence_unique_ids() {
        let a = MetalFence::new("a");
        let b = MetalFence::new("b");
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn fence_display() {
        let f = MetalFence::new("sync");
        let s = format!("{f}");
        assert!(s.contains("MetalFence"));
        assert!(s.contains("\"sync\""));
    }

    // ── MetalCaptureScope ───────────────────────────────────────────────

    #[test]
    fn capture_initially_inactive() {
        let c = MetalCaptureScope::new("debug");
        assert!(!c.is_active());
    }

    #[test]
    fn capture_begin_activates() {
        let mut c = MetalCaptureScope::new("debug");
        c.begin();
        assert!(c.is_active());
    }

    #[test]
    fn capture_end_deactivates() {
        let mut c = MetalCaptureScope::new("debug");
        c.begin();
        c.end();
        assert!(!c.is_active());
    }

    #[test]
    fn capture_frame_count() {
        let mut c = MetalCaptureScope::new("debug");
        c.begin();
        c.end();
        c.begin();
        c.end();
        assert_eq!(c.frame_count(), 2);
    }

    #[test]
    fn capture_label() {
        let c = MetalCaptureScope::new("xcode");
        assert_eq!(c.label(), "xcode");
    }

    #[test]
    #[should_panic(expected = "capture frame already active")]
    fn capture_double_begin_panics() {
        let mut c = MetalCaptureScope::new("debug");
        c.begin();
        c.begin();
    }

    #[test]
    #[should_panic(expected = "no capture frame active")]
    fn capture_end_without_begin_panics() {
        let mut c = MetalCaptureScope::new("debug");
        c.end();
    }

    #[test]
    fn capture_display() {
        let c = MetalCaptureScope::new("scope");
        let s = format!("{c}");
        assert!(s.contains("MetalCaptureScope"));
        assert!(s.contains("\"scope\""));
    }

    // ── MetalBackend ────────────────────────────────────────────────────

    #[test]
    fn backend_new_from_default_config() {
        let b = MetalBackend::new(sample_config());
        assert_eq!(b.device().index(), 0);
        assert_eq!(b.dispatch_count(), 0);
    }

    #[test]
    fn backend_device_name() {
        let b = MetalBackend::new(sample_config());
        assert_eq!(b.device().name(), "CPU Reference Device");
    }

    #[test]
    fn backend_queue_capacity() {
        let b = MetalBackend::new(sample_config());
        assert_eq!(b.queue().capacity(), 3);
    }

    #[test]
    fn backend_no_capture_by_default() {
        let b = MetalBackend::new(sample_config());
        assert!(b.capture_scope().is_none());
    }

    #[test]
    fn backend_capture_when_enabled() {
        let b = MetalBackend::new(capture_config());
        assert!(b.capture_scope().is_some());
    }

    #[test]
    fn backend_add_library() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        let lib = b.get_library("ops").unwrap();
        assert_eq!(lib.function_names().len(), 2);
    }

    #[test]
    fn backend_add_precompiled_library() {
        let mut b = MetalBackend::new(sample_config());
        b.add_precompiled_library("pre", "lib.metallib", vec!["kern".to_string()]);
        assert!(b.get_library("pre").is_some());
    }

    #[test]
    fn backend_create_pipeline() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        let p = b.create_pipeline("add_arrays");
        assert!(p.is_some());
        assert_eq!(p.unwrap().function_name(), "add_arrays");
    }

    #[test]
    fn backend_create_pipeline_missing_function() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        assert!(b.create_pipeline("no_such_function").is_none());
    }

    #[test]
    fn backend_get_pipeline() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("add_arrays");
        assert!(b.get_pipeline("add_arrays").is_some());
    }

    #[test]
    fn backend_dispatch_success() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("add_arrays");

        let result = b.dispatch("add_arrays", DispatchSize::one_d(1, 256));
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.success);
        assert_eq!(r.total_threads, 256);
    }

    #[test]
    fn backend_dispatch_increments_count() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("add_arrays");
        b.dispatch("add_arrays", DispatchSize::one_d(1, 64));
        b.dispatch("add_arrays", DispatchSize::one_d(1, 64));
        assert_eq!(b.dispatch_count(), 2);
    }

    #[test]
    fn backend_dispatch_no_pipeline_returns_none() {
        let mut b = MetalBackend::new(sample_config());
        let result = b.dispatch("nope", DispatchSize::one_d(1, 1));
        assert!(result.is_none());
    }

    #[test]
    fn backend_dispatch_no_slot_returns_none() {
        let cfg = MetalConfig::new().with_command_buffer_count(1);
        let mut b = MetalBackend::new(cfg);
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("add_arrays");

        // Occupy the only slot manually
        b.queue_mut().acquire();

        let result = b.dispatch("add_arrays", DispatchSize::one_d(1, 1));
        assert!(result.is_none());
    }

    #[test]
    fn backend_reset_clears_state() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("add_arrays");
        b.dispatch("add_arrays", DispatchSize::one_d(1, 1));
        b.reset();
        assert_eq!(b.dispatch_count(), 0);
        assert!(b.get_pipeline("add_arrays").is_none());
    }

    #[test]
    fn backend_config_ref() {
        let cfg = capture_config();
        let b = MetalBackend::new(cfg.clone());
        assert_eq!(b.config(), &cfg);
    }

    #[test]
    fn backend_capture_scope_mut() {
        let mut b = MetalBackend::new(capture_config());
        let scope = b.capture_scope_mut().unwrap();
        scope.begin();
        scope.end();
        assert_eq!(b.capture_scope().unwrap().frame_count(), 1);
    }

    #[test]
    fn backend_display() {
        let b = MetalBackend::new(sample_config());
        let s = format!("{b}");
        assert!(s.contains("MetalBackend"));
        assert!(s.contains("CPU Reference Device"));
    }

    #[test]
    fn backend_multiple_libraries() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("a", "kernel void ka(uint i) {}");
        b.add_library_from_source("b", "kernel void kb(uint i) {}");
        assert!(b.get_library("a").is_some());
        assert!(b.get_library("b").is_some());
    }

    #[test]
    fn backend_pipeline_from_second_library() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("a", "kernel void ka(uint i) {}");
        b.add_library_from_source("b", "kernel void kb(uint i) {}");
        assert!(b.create_pipeline("kb").is_some());
    }

    #[test]
    fn backend_dispatch_result_message() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("scale_array");
        let r = b.dispatch("scale_array", DispatchSize::one_d(2, 128)).unwrap();
        assert!(r.message.contains("256 threads"));
        assert_eq!(r.function_name, "scale_array");
    }

    #[test]
    fn backend_dispatch_2d() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.create_pipeline("add_arrays");
        let r = b.dispatch("add_arrays", DispatchSize::two_d(2, 2, 8, 8)).unwrap();
        assert_eq!(r.total_threads, 256);
    }

    #[test]
    fn backend_reset_preserves_libraries() {
        let mut b = MetalBackend::new(sample_config());
        b.add_library_from_source("ops", SAMPLE_MSL);
        b.reset();
        assert!(b.get_library("ops").is_some());
    }
}
