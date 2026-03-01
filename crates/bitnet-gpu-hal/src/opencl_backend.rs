//! `OpenCL` backend with platform/device/context/queue/kernel management.
//!
//! This is a CPU reference implementation that simulates `OpenCL` behavior for
//! development and testing without requiring an `OpenCL` runtime. All types
//! expose the full API surface; GPU-requiring paths are marked with comments
//! indicating where real `OpenCL` calls would be dispatched.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur in the `OpenCL` backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenCLError {
    PlatformNotFound(u32),
    DeviceNotFound(u32),
    InvalidBufferSize,
    BufferOutOfRange { offset: usize, len: usize, capacity: usize },
    KernelCompilationFailed(String),
    EmptyKernelSource,
    InvalidNDRange(String),
    InvalidArgIndex { index: u32, max: u32 },
    ContextNotInitialized,
    ProgramNotBuilt,
    BackendNotReady(String),
}

impl fmt::Display for OpenCLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlatformNotFound(i) => write!(f, "platform index {i} not found"),
            Self::DeviceNotFound(i) => write!(f, "device index {i} not found"),
            Self::InvalidBufferSize => write!(f, "buffer size must be > 0"),
            Self::BufferOutOfRange { offset, len, capacity } => {
                write!(
                    f,
                    "buffer range [{offset}..{end}] exceeds capacity {capacity}",
                    end = offset + len
                )
            }
            Self::KernelCompilationFailed(msg) => write!(f, "kernel compilation failed: {msg}"),
            Self::EmptyKernelSource => write!(f, "kernel source is empty"),
            Self::InvalidNDRange(msg) => write!(f, "invalid NDRange: {msg}"),
            Self::InvalidArgIndex { index, max } => {
                write!(f, "arg index {index} exceeds max {max}")
            }
            Self::ContextNotInitialized => write!(f, "OpenCL context not initialized"),
            Self::ProgramNotBuilt => write!(f, "program has not been built"),
            Self::BackendNotReady(msg) => write!(f, "backend not ready: {msg}"),
        }
    }
}

impl std::error::Error for OpenCLError {}

pub type Result<T> = std::result::Result<T, OpenCLError>;

// ---------------------------------------------------------------------------
// OpenCLConfig
// ---------------------------------------------------------------------------

/// Configuration for initializing the `OpenCL` backend.
#[derive(Debug, Clone)]
pub struct OpenCLConfig {
    pub platform_index: u32,
    pub device_index: u32,
    pub profiling_enabled: bool,
    pub queue_count: u32,
    pub kernel_cache_dir: Option<PathBuf>,
}

impl Default for OpenCLConfig {
    fn default() -> Self {
        Self {
            platform_index: 0,
            device_index: 0,
            profiling_enabled: false,
            queue_count: 1,
            kernel_cache_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenCLPlatformInfo
// ---------------------------------------------------------------------------

/// Information about an available `OpenCL` platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCLPlatformInfo {
    pub name: String,
    pub vendor: String,
    pub version: String,
    pub extensions: Vec<String>,
}

impl OpenCLPlatformInfo {
    /// Returns a mock platform simulating an Intel `OpenCL` CPU runtime.
    pub fn mock_cpu_platform() -> Self {
        Self {
            name: "Intel(R) OpenCL".into(),
            vendor: "Intel(R) Corporation".into(),
            version: "OpenCL 3.0".into(),
            extensions: vec![
                "cl_khr_fp16".into(),
                "cl_khr_fp64".into(),
                "cl_intel_subgroups".into(),
            ],
        }
    }

    /// Returns a mock platform simulating an Intel Arc GPU runtime.
    pub fn mock_gpu_platform() -> Self {
        Self {
            name: "Intel(R) OpenCL Graphics".into(),
            vendor: "Intel(R) Corporation".into(),
            version: "OpenCL 3.0".into(),
            extensions: vec![
                "cl_khr_fp16".into(),
                "cl_intel_subgroups".into(),
                "cl_intel_required_subgroup_size".into(),
            ],
        }
    }

    pub fn has_extension(&self, ext: &str) -> bool {
        self.extensions.iter().any(|e| e == ext)
    }
}

// ---------------------------------------------------------------------------
// OpenCLDeviceInfo
// ---------------------------------------------------------------------------

/// Detailed information about an `OpenCL` device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenCLDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub compute_units: u32,
    pub max_work_group_size: usize,
    pub global_memory_bytes: u64,
    pub local_memory_bytes: u64,
    pub supports_fp16: bool,
    pub max_clock_frequency_mhz: u32,
    pub device_type: OpenCLDeviceType,
}

/// Type of `OpenCL` device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenCLDeviceType {
    Cpu,
    Gpu,
    Accelerator,
}

impl fmt::Display for OpenCLDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Accelerator => write!(f, "Accelerator"),
        }
    }
}

impl OpenCLDeviceInfo {
    /// Returns a mock CPU device.
    pub fn mock_cpu() -> Self {
        Self {
            name: "Simulated CPU Device".into(),
            vendor: "BitNet Reference".into(),
            compute_units: 8,
            max_work_group_size: 1024,
            global_memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
            local_memory_bytes: 64 * 1024,
            supports_fp16: true,
            max_clock_frequency_mhz: 3600,
            device_type: OpenCLDeviceType::Cpu,
        }
    }

    /// Returns a mock GPU device resembling Intel Arc.
    pub fn mock_gpu() -> Self {
        Self {
            name: "Simulated Intel Arc GPU".into(),
            vendor: "Intel(R) Corporation".into(),
            compute_units: 512,
            max_work_group_size: 1024,
            global_memory_bytes: 16 * 1024 * 1024 * 1024,
            local_memory_bytes: 128 * 1024,
            supports_fp16: true,
            max_clock_frequency_mhz: 2400,
            device_type: OpenCLDeviceType::Gpu,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenCLContext
// ---------------------------------------------------------------------------

/// Wraps an `OpenCL` context handle and manages its lifetime.
///
/// In this CPU reference implementation the "context" simply holds the
/// selected platform/device info. A real implementation would store the
/// `cl_context` handle here.
#[derive(Debug, Clone)]
pub struct OpenCLContext {
    platform: OpenCLPlatformInfo,
    device: OpenCLDeviceInfo,
    /// Opaque handle – would be `cl_context` in a real build.
    _handle: u64,
}

impl OpenCLContext {
    /// Creates a new simulated context for the given platform and device.
    pub const fn new(platform: OpenCLPlatformInfo, device: OpenCLDeviceInfo) -> Self {
        // Real: clCreateContext(...)
        Self { platform, device, _handle: 0xDEAD_BEEF }
    }

    pub const fn platform_info(&self) -> &OpenCLPlatformInfo {
        &self.platform
    }

    pub const fn device_info(&self) -> &OpenCLDeviceInfo {
        &self.device
    }

    pub const fn supports_fp16(&self) -> bool {
        self.device.supports_fp16
    }

    pub const fn compute_units(&self) -> u32 {
        self.device.compute_units
    }

    pub const fn global_memory_bytes(&self) -> u64 {
        self.device.global_memory_bytes
    }

    pub const fn local_memory_bytes(&self) -> u64 {
        self.device.local_memory_bytes
    }
}

// ---------------------------------------------------------------------------
// ExecutionMode
// ---------------------------------------------------------------------------

/// Queue execution ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    InOrder,
    OutOfOrder,
}

// ---------------------------------------------------------------------------
// OpenCLCommandQueue
// ---------------------------------------------------------------------------

/// Wraps an `OpenCL` command queue.
///
/// Real implementation would hold a `cl_command_queue` handle.
#[derive(Debug)]
pub struct OpenCLCommandQueue {
    mode: ExecutionMode,
    profiling: bool,
    /// Number of commands enqueued (for simulation bookkeeping).
    enqueued_count: u64,
}

impl OpenCLCommandQueue {
    /// Creates a new command queue bound to the given context.
    pub const fn new(_ctx: &OpenCLContext, mode: ExecutionMode, profiling: bool) -> Self {
        // Real: clCreateCommandQueue / clCreateCommandQueueWithProperties
        Self { mode, profiling, enqueued_count: 0 }
    }

    pub const fn execution_mode(&self) -> ExecutionMode {
        self.mode
    }

    pub const fn profiling_enabled(&self) -> bool {
        self.profiling
    }

    pub const fn enqueued_count(&self) -> u64 {
        self.enqueued_count
    }

    /// Simulate flushing the command queue.
    pub const fn flush(&mut self) {
        // Real: clFlush(queue)
    }

    /// Simulate finishing (blocking until all commands complete).
    pub const fn finish(&mut self) {
        // Real: clFinish(queue)
    }

    /// Increment the internal enqueued counter (used by launcher).
    const fn record_enqueue(&mut self) {
        self.enqueued_count += 1;
    }
}

// ---------------------------------------------------------------------------
// OpenCLBuffer<T>
// ---------------------------------------------------------------------------

/// Typed GPU buffer backed by a CPU `Vec<T>` in this reference build.
///
/// A real implementation would wrap `cl_mem` and issue
/// `clEnqueueReadBuffer` / `clEnqueueWriteBuffer` calls.
#[derive(Debug, Clone)]
pub struct OpenCLBuffer<T: Clone + Default> {
    data: Vec<T>,
    len: usize,
}

impl<T: Clone + Default> OpenCLBuffer<T> {
    /// Allocate a buffer of `len` elements, zero-initialized.
    pub fn alloc(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(OpenCLError::InvalidBufferSize);
        }
        // Real: clCreateBuffer(CL_MEM_READ_WRITE, len * sizeof(T), ...)
        Ok(Self { data: vec![T::default(); len], len })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes (assuming `std::mem::size_of::<T>()` per element).
    pub const fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Write `src` into the buffer starting at `offset`.
    pub fn write(&mut self, offset: usize, src: &[T]) -> Result<()> {
        if offset + src.len() > self.len {
            return Err(OpenCLError::BufferOutOfRange {
                offset,
                len: src.len(),
                capacity: self.len,
            });
        }
        // Real: clEnqueueWriteBuffer(queue, mem, CL_TRUE, offset, ...)
        self.data[offset..offset + src.len()].clone_from_slice(src);
        Ok(())
    }

    /// Read `len` elements starting at `offset`.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<T>> {
        if offset + len > self.len {
            return Err(OpenCLError::BufferOutOfRange { offset, len, capacity: self.len });
        }
        // Real: clEnqueueReadBuffer(queue, mem, CL_TRUE, offset, ...)
        Ok(self.data[offset..offset + len].to_vec())
    }

    /// Read entire buffer contents.
    pub fn read_all(&self) -> Vec<T> {
        self.data.clone()
    }

    /// Copy `len` elements from `src` (at `src_offset`) into `self` (at
    /// `dst_offset`).
    pub fn copy_from(
        &mut self,
        src: &Self,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        if src_offset + len > src.len {
            return Err(OpenCLError::BufferOutOfRange {
                offset: src_offset,
                len,
                capacity: src.len,
            });
        }
        if dst_offset + len > self.len {
            return Err(OpenCLError::BufferOutOfRange {
                offset: dst_offset,
                len,
                capacity: self.len,
            });
        }
        // Real: clEnqueueCopyBuffer(queue, src_mem, dst_mem, ...)
        let slice = src.data[src_offset..src_offset + len].to_vec();
        self.data[dst_offset..dst_offset + len].clone_from_slice(&slice);
        Ok(())
    }

    /// Fill the entire buffer with `value`.
    pub fn fill(&mut self, value: T) {
        // Real: clEnqueueFillBuffer(queue, mem, &value, ...)
        self.data.fill(value);
    }
}

// ---------------------------------------------------------------------------
// OpenCLKernelSource
// ---------------------------------------------------------------------------

/// Manages `.cl` source code, compilation options, and binary caching.
#[derive(Debug, Clone)]
pub struct OpenCLKernelSource {
    source: String,
    build_options: String,
    build_log: Option<String>,
    compiled: bool,
    /// Cached binary (simulated).
    cached_binary: Option<Vec<u8>>,
}

impl OpenCLKernelSource {
    /// Create a new kernel source from `OpenCL` C code.
    pub fn from_source(source: &str) -> Result<Self> {
        if source.is_empty() {
            return Err(OpenCLError::EmptyKernelSource);
        }
        Ok(Self {
            source: source.to_string(),
            build_options: String::new(),
            build_log: None,
            compiled: false,
            cached_binary: None,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn set_build_options(&mut self, opts: &str) {
        self.build_options = opts.to_string();
    }

    pub fn build_options(&self) -> &str {
        &self.build_options
    }

    pub fn build_log(&self) -> Option<&str> {
        self.build_log.as_deref()
    }

    pub const fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Simulate compilation. Returns an error if source contains `#error`.
    pub fn compile(&mut self) -> Result<()> {
        // Real: clBuildProgram(program, ...)
        if self.source.contains("#error") {
            let msg = "source contains #error directive".to_string();
            self.build_log = Some(msg.clone());
            return Err(OpenCLError::KernelCompilationFailed(msg));
        }
        self.build_log = Some("Build successful".into());
        self.compiled = true;
        // Simulate binary caching.
        self.cached_binary = Some(self.source.as_bytes().to_vec());
        Ok(())
    }

    /// Load a cached binary (simulated).
    pub fn load_cached_binary(&mut self, binary: Vec<u8>) {
        self.cached_binary = Some(binary);
        self.compiled = true;
        self.build_log = Some("Loaded from cache".into());
    }

    pub fn cached_binary(&self) -> Option<&[u8]> {
        self.cached_binary.as_deref()
    }

    /// Save binary to a cache directory (simulated – returns the path that
    /// *would* be written).
    pub fn save_to_cache(&self, cache_dir: &Path, name: &str) -> Option<PathBuf> {
        if self.cached_binary.is_some() {
            Some(cache_dir.join(format!("{name}.clbin")))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// ProgramBuildSource
// ---------------------------------------------------------------------------

/// Source type for building an `OpenCLProgram`.
#[derive(Debug, Clone)]
pub enum ProgramBuildSource {
    /// `OpenCL` C source code.
    Source(String),
    /// SPIR-V binary.
    SpirV(Vec<u8>),
}

// ---------------------------------------------------------------------------
// OpenCLProgram
// ---------------------------------------------------------------------------

/// Wraps a compiled `OpenCL` program. Supports both source and SPIR-V paths.
#[derive(Debug, Clone)]
pub struct OpenCLProgram {
    build_source: ProgramBuildSource,
    kernel_names: Vec<String>,
    built: bool,
    build_log: Option<String>,
    build_options: String,
}

impl OpenCLProgram {
    /// Create a program from `OpenCL` C source.
    pub fn from_source(source: &str) -> Result<Self> {
        if source.is_empty() {
            return Err(OpenCLError::EmptyKernelSource);
        }
        Ok(Self {
            build_source: ProgramBuildSource::Source(source.to_string()),
            kernel_names: Vec::new(),
            built: false,
            build_log: None,
            build_options: String::new(),
        })
    }

    /// Create a program from SPIR-V binary.
    pub fn from_spirv(binary: Vec<u8>) -> Result<Self> {
        if binary.is_empty() {
            return Err(OpenCLError::EmptyKernelSource);
        }
        Ok(Self {
            build_source: ProgramBuildSource::SpirV(binary),
            kernel_names: Vec::new(),
            built: false,
            build_log: None,
            build_options: String::new(),
        })
    }

    pub fn set_build_options(&mut self, opts: &str) {
        self.build_options = opts.to_string();
    }

    pub fn build_options(&self) -> &str {
        &self.build_options
    }

    /// Build the program. Extracts kernel names from `__kernel` annotations
    /// in source mode or uses placeholder names for SPIR-V.
    pub fn build(&mut self, _ctx: &OpenCLContext) -> Result<()> {
        // Real: clBuildProgram / clCreateProgramWithIL
        match &self.build_source {
            ProgramBuildSource::Source(src) => {
                if src.contains("#error") {
                    let msg = "compilation error in source".to_string();
                    self.build_log = Some(msg.clone());
                    return Err(OpenCLError::KernelCompilationFailed(msg));
                }
                // Extract __kernel function names (simplistic parser).
                self.kernel_names = src
                    .split("__kernel")
                    .skip(1)
                    .filter_map(|fragment| {
                        // Expect `void name(` pattern after __kernel.
                        let trimmed = fragment.trim();
                        let after_void = trimmed.strip_prefix("void ")?;
                        let name = after_void.split('(').next()?;
                        let name = name.trim();
                        if name.is_empty() { None } else { Some(name.to_string()) }
                    })
                    .collect();
            }
            ProgramBuildSource::SpirV(_) => {
                self.kernel_names = vec!["spirv_entry".into()];
            }
        }
        self.built = true;
        self.build_log = Some("Build successful".into());
        Ok(())
    }

    pub const fn is_built(&self) -> bool {
        self.built
    }

    pub fn build_log(&self) -> Option<&str> {
        self.build_log.as_deref()
    }

    pub fn kernel_names(&self) -> &[String] {
        &self.kernel_names
    }
}

// ---------------------------------------------------------------------------
// NDRange
// ---------------------------------------------------------------------------

/// N-dimensional execution range for kernel launches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NDRange {
    pub global: Vec<usize>,
    pub local: Option<Vec<usize>>,
    pub offset: Option<Vec<usize>>,
}

impl NDRange {
    /// 1-D range.
    pub fn new_1d(global: usize) -> Self {
        Self { global: vec![global], local: None, offset: None }
    }

    /// 2-D range.
    pub fn new_2d(gx: usize, gy: usize) -> Self {
        Self { global: vec![gx, gy], local: None, offset: None }
    }

    /// 3-D range.
    pub fn new_3d(gx: usize, gy: usize, gz: usize) -> Self {
        Self { global: vec![gx, gy, gz], local: None, offset: None }
    }

    pub fn with_local(mut self, local: Vec<usize>) -> Result<Self> {
        if local.len() != self.global.len() {
            return Err(OpenCLError::InvalidNDRange("local dims must match global dims".into()));
        }
        for (g, l) in self.global.iter().zip(local.iter()) {
            if *l == 0 {
                return Err(OpenCLError::InvalidNDRange("local work size must be > 0".into()));
            }
            if g % l != 0 {
                return Err(OpenCLError::InvalidNDRange(format!(
                    "global size {g} not divisible by local size {l}"
                )));
            }
        }
        self.local = Some(local);
        Ok(self)
    }

    pub fn with_offset(mut self, offset: Vec<usize>) -> Result<Self> {
        if offset.len() != self.global.len() {
            return Err(OpenCLError::InvalidNDRange("offset dims must match global dims".into()));
        }
        self.offset = Some(offset);
        Ok(self)
    }

    pub const fn dimensions(&self) -> usize {
        self.global.len()
    }

    /// Total number of work items.
    pub fn total_work_items(&self) -> usize {
        self.global.iter().product()
    }

    /// Number of work groups (if local size is set).
    pub fn num_work_groups(&self) -> Option<usize> {
        self.local
            .as_ref()
            .map(|l| self.global.iter().zip(l.iter()).map(|(g, loc)| g / loc).product())
    }

    /// Validate the configuration.
    pub fn validate(&self, max_work_group_size: usize) -> Result<()> {
        if self.global.is_empty() || self.global.len() > 3 {
            return Err(OpenCLError::InvalidNDRange("dimensions must be 1, 2, or 3".into()));
        }
        for &g in &self.global {
            if g == 0 {
                return Err(OpenCLError::InvalidNDRange("global work size must be > 0".into()));
            }
        }
        if let Some(ref local) = self.local {
            let total_local: usize = local.iter().product();
            if total_local > max_work_group_size {
                return Err(OpenCLError::InvalidNDRange(format!(
                    "local work size {total_local} exceeds max {max_work_group_size}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KernelArg
// ---------------------------------------------------------------------------

/// Represents a kernel argument value (type-erased for storage).
#[derive(Debug, Clone)]
pub enum KernelArg {
    Int(i32),
    UInt(u32),
    Float(f32),
    Long(i64),
    ULong(u64),
    /// Represents a buffer handle (index into some buffer table).
    BufferHandle(u64),
    LocalMemory(usize),
}

// ---------------------------------------------------------------------------
// OpenCLKernelLauncher
// ---------------------------------------------------------------------------

/// Sets kernel arguments, configures `NDRange`, and enqueues execution.
#[derive(Debug)]
pub struct OpenCLKernelLauncher {
    kernel_name: String,
    args: HashMap<u32, KernelArg>,
    max_args: u32,
    nd_range: Option<NDRange>,
    launch_count: u64,
}

impl OpenCLKernelLauncher {
    pub fn new(kernel_name: &str, max_args: u32) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            args: HashMap::new(),
            max_args,
            nd_range: None,
            launch_count: 0,
        }
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn set_arg(&mut self, index: u32, arg: KernelArg) -> Result<()> {
        if index >= self.max_args {
            return Err(OpenCLError::InvalidArgIndex { index, max: self.max_args });
        }
        // Real: clSetKernelArg(kernel, index, ...)
        self.args.insert(index, arg);
        Ok(())
    }

    pub fn get_arg(&self, index: u32) -> Option<&KernelArg> {
        self.args.get(&index)
    }

    pub fn arg_count(&self) -> usize {
        self.args.len()
    }

    pub fn set_nd_range(&mut self, range: NDRange) {
        self.nd_range = Some(range);
    }

    pub const fn nd_range(&self) -> Option<&NDRange> {
        self.nd_range.as_ref()
    }

    /// Enqueue the kernel for execution on the given queue.
    pub fn enqueue(&mut self, queue: &mut OpenCLCommandQueue) -> Result<()> {
        let range = self
            .nd_range
            .as_ref()
            .ok_or_else(|| OpenCLError::InvalidNDRange("no NDRange configured".into()))?;
        range.validate(1024)?;
        // Real: clEnqueueNDRangeKernel(queue, kernel, dims, ...)
        queue.record_enqueue();
        self.launch_count += 1;
        Ok(())
    }

    pub const fn launch_count(&self) -> u64 {
        self.launch_count
    }

    /// Reset arguments (keeps kernel name and `NDRange`).
    pub fn clear_args(&mut self) {
        self.args.clear();
    }
}

// ---------------------------------------------------------------------------
// OpenCLBackend
// ---------------------------------------------------------------------------

/// Top-level orchestrator: detect platforms → select device → create context →
/// compile kernels → execute.
#[derive(Debug)]
pub struct OpenCLBackend {
    config: OpenCLConfig,
    platforms: Vec<OpenCLPlatformInfo>,
    devices: Vec<OpenCLDeviceInfo>,
    context: Option<OpenCLContext>,
    queues: Vec<OpenCLCommandQueue>,
    programs: HashMap<String, OpenCLProgram>,
    ready: bool,
}

impl OpenCLBackend {
    /// Enumerate available platforms and devices (simulated).
    pub fn new(config: OpenCLConfig) -> Self {
        // Real: clGetPlatformIDs / clGetDeviceIDs
        let platforms =
            vec![OpenCLPlatformInfo::mock_cpu_platform(), OpenCLPlatformInfo::mock_gpu_platform()];
        let devices = vec![OpenCLDeviceInfo::mock_cpu(), OpenCLDeviceInfo::mock_gpu()];
        Self {
            config,
            platforms,
            devices,
            context: None,
            queues: Vec::new(),
            programs: HashMap::new(),
            ready: false,
        }
    }

    pub fn platforms(&self) -> &[OpenCLPlatformInfo] {
        &self.platforms
    }

    pub fn devices(&self) -> &[OpenCLDeviceInfo] {
        &self.devices
    }

    /// Select device by index from the device list.
    pub fn select_device(&self, index: u32) -> Result<&OpenCLDeviceInfo> {
        self.devices.get(index as usize).ok_or(OpenCLError::DeviceNotFound(index))
    }

    /// Select first device matching the given type.
    pub fn select_device_by_type(
        &self,
        device_type: OpenCLDeviceType,
    ) -> Result<(u32, &OpenCLDeviceInfo)> {
        self.devices
            .iter()
            .enumerate()
            .find(|(_, d)| d.device_type == device_type)
            .map(|(i, d)| (u32::try_from(i).unwrap_or(u32::MAX), d))
            .ok_or(OpenCLError::DeviceNotFound(u32::MAX))
    }

    /// Initialise the backend: create context and command queues.
    pub fn initialize(&mut self) -> Result<()> {
        let pi = self.config.platform_index as usize;
        let di = self.config.device_index as usize;
        let platform = self
            .platforms
            .get(pi)
            .ok_or(OpenCLError::PlatformNotFound(self.config.platform_index))?
            .clone();
        let device = self
            .devices
            .get(di)
            .ok_or(OpenCLError::DeviceNotFound(self.config.device_index))?
            .clone();

        let ctx = OpenCLContext::new(platform, device);

        let mode = ExecutionMode::InOrder;
        for _ in 0..self.config.queue_count.max(1) {
            self.queues.push(OpenCLCommandQueue::new(&ctx, mode, self.config.profiling_enabled));
        }
        self.context = Some(ctx);
        self.ready = true;
        Ok(())
    }

    pub const fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn context(&self) -> Result<&OpenCLContext> {
        self.context.as_ref().ok_or(OpenCLError::ContextNotInitialized)
    }

    pub fn queue(&mut self, index: usize) -> Option<&mut OpenCLCommandQueue> {
        self.queues.get_mut(index)
    }

    pub const fn queue_count(&self) -> usize {
        self.queues.len()
    }

    /// Build and register a program from source.
    pub fn build_program(&mut self, name: &str, source: &str) -> Result<()> {
        let ctx = self.context.as_ref().ok_or(OpenCLError::ContextNotInitialized)?;
        let mut program = OpenCLProgram::from_source(source)?;
        program.build(ctx)?;
        self.programs.insert(name.to_string(), program);
        Ok(())
    }

    /// Build and register a program from SPIR-V.
    pub fn build_program_spirv(&mut self, name: &str, binary: Vec<u8>) -> Result<()> {
        let ctx = self.context.as_ref().ok_or(OpenCLError::ContextNotInitialized)?;
        let mut program = OpenCLProgram::from_spirv(binary)?;
        program.build(ctx)?;
        self.programs.insert(name.to_string(), program);
        Ok(())
    }

    pub fn get_program(&self, name: &str) -> Option<&OpenCLProgram> {
        self.programs.get(name)
    }

    /// Create a kernel launcher for a named kernel. The kernel must belong to
    /// a registered program.
    pub fn create_launcher(
        &self,
        program_name: &str,
        kernel_name: &str,
        max_args: u32,
    ) -> Result<OpenCLKernelLauncher> {
        let program = self.programs.get(program_name).ok_or(OpenCLError::ProgramNotBuilt)?;
        if !program.is_built() {
            return Err(OpenCLError::ProgramNotBuilt);
        }
        Ok(OpenCLKernelLauncher::new(kernel_name, max_args))
    }

    /// Convenience: allocate a typed buffer (delegates to `OpenCLBuffer`).
    pub fn alloc_buffer<T: Clone + Default>(&self, len: usize) -> Result<OpenCLBuffer<T>> {
        if !self.ready {
            return Err(OpenCLError::BackendNotReady("call initialize() first".into()));
        }
        OpenCLBuffer::alloc(len)
    }

    /// Shut down the backend, releasing resources.
    pub fn shutdown(&mut self) {
        // Real: clReleaseCommandQueue, clReleaseContext, etc.
        self.queues.clear();
        self.programs.clear();
        self.context = None;
        self.ready = false;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Platform tests ---------------------------------------------------

    #[test]
    fn test_mock_cpu_platform_fields() {
        let p = OpenCLPlatformInfo::mock_cpu_platform();
        assert_eq!(p.name, "Intel(R) OpenCL");
        assert_eq!(p.vendor, "Intel(R) Corporation");
        assert!(p.version.contains("3.0"));
    }

    #[test]
    fn test_mock_gpu_platform_fields() {
        let p = OpenCLPlatformInfo::mock_gpu_platform();
        assert!(p.name.contains("Graphics"));
    }

    #[test]
    fn test_platform_has_extension() {
        let p = OpenCLPlatformInfo::mock_cpu_platform();
        assert!(p.has_extension("cl_khr_fp16"));
        assert!(!p.has_extension("cl_khr_gl_sharing"));
    }

    #[test]
    fn test_platform_extensions_list() {
        let p = OpenCLPlatformInfo::mock_cpu_platform();
        assert!(p.extensions.len() >= 2);
    }

    #[test]
    fn test_gpu_platform_intel_subgroups() {
        let p = OpenCLPlatformInfo::mock_gpu_platform();
        assert!(p.has_extension("cl_intel_subgroups"));
    }

    // -- Device tests -----------------------------------------------------

    #[test]
    fn test_mock_cpu_device_type() {
        let d = OpenCLDeviceInfo::mock_cpu();
        assert_eq!(d.device_type, OpenCLDeviceType::Cpu);
    }

    #[test]
    fn test_mock_gpu_device_type() {
        let d = OpenCLDeviceInfo::mock_gpu();
        assert_eq!(d.device_type, OpenCLDeviceType::Gpu);
    }

    #[test]
    fn test_device_compute_units() {
        let d = OpenCLDeviceInfo::mock_gpu();
        assert!(d.compute_units > 0);
        assert_eq!(d.compute_units, 512);
    }

    #[test]
    fn test_device_memory() {
        let d = OpenCLDeviceInfo::mock_gpu();
        assert!(d.global_memory_bytes > 0);
        assert!(d.local_memory_bytes > 0);
    }

    #[test]
    fn test_device_fp16_support() {
        assert!(OpenCLDeviceInfo::mock_gpu().supports_fp16);
        assert!(OpenCLDeviceInfo::mock_cpu().supports_fp16);
    }

    #[test]
    fn test_device_max_work_group_size() {
        let d = OpenCLDeviceInfo::mock_cpu();
        assert_eq!(d.max_work_group_size, 1024);
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(format!("{}", OpenCLDeviceType::Cpu), "CPU");
        assert_eq!(format!("{}", OpenCLDeviceType::Gpu), "GPU");
        assert_eq!(format!("{}", OpenCLDeviceType::Accelerator), "Accelerator");
    }

    #[test]
    fn test_device_clock_frequency() {
        let d = OpenCLDeviceInfo::mock_gpu();
        assert!(d.max_clock_frequency_mhz > 0);
    }

    // -- Context tests ----------------------------------------------------

    #[test]
    fn test_context_creation() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        assert!(ctx.supports_fp16());
    }

    #[test]
    fn test_context_device_info() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_gpu_platform(),
            OpenCLDeviceInfo::mock_gpu(),
        );
        assert_eq!(ctx.compute_units(), 512);
        assert!(ctx.global_memory_bytes() > 0);
        assert!(ctx.local_memory_bytes() > 0);
    }

    #[test]
    fn test_context_platform_info() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        assert_eq!(ctx.platform_info().vendor, "Intel(R) Corporation");
    }

    #[test]
    fn test_context_device_name() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        assert!(ctx.device_info().name.contains("Simulated"));
    }

    // -- Queue tests ------------------------------------------------------

    #[test]
    fn test_queue_in_order() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let q = OpenCLCommandQueue::new(&ctx, ExecutionMode::InOrder, false);
        assert_eq!(q.execution_mode(), ExecutionMode::InOrder);
        assert!(!q.profiling_enabled());
    }

    #[test]
    fn test_queue_out_of_order() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let q = OpenCLCommandQueue::new(&ctx, ExecutionMode::OutOfOrder, true);
        assert_eq!(q.execution_mode(), ExecutionMode::OutOfOrder);
        assert!(q.profiling_enabled());
    }

    #[test]
    fn test_queue_enqueue_count() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut q = OpenCLCommandQueue::new(&ctx, ExecutionMode::InOrder, false);
        assert_eq!(q.enqueued_count(), 0);
        q.record_enqueue();
        assert_eq!(q.enqueued_count(), 1);
    }

    #[test]
    fn test_queue_flush_finish() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut q = OpenCLCommandQueue::new(&ctx, ExecutionMode::InOrder, false);
        q.flush();
        q.finish();
        // No panic = success for simulated ops.
    }

    // -- Buffer tests -----------------------------------------------------

    #[test]
    fn test_buffer_alloc() {
        let buf = OpenCLBuffer::<f32>::alloc(128).unwrap();
        assert_eq!(buf.len(), 128);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_zero_size() {
        let err = OpenCLBuffer::<f32>::alloc(0).unwrap_err();
        assert_eq!(err, OpenCLError::InvalidBufferSize);
    }

    #[test]
    fn test_buffer_write_read() {
        let mut buf = OpenCLBuffer::<i32>::alloc(10).unwrap();
        buf.write(0, &[1, 2, 3, 4, 5]).unwrap();
        let data = buf.read(0, 5).unwrap();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_buffer_write_offset() {
        let mut buf = OpenCLBuffer::<u8>::alloc(8).unwrap();
        buf.write(4, &[0xAA, 0xBB]).unwrap();
        let all = buf.read_all();
        assert_eq!(all[4], 0xAA);
        assert_eq!(all[5], 0xBB);
        assert_eq!(all[0], 0);
    }

    #[test]
    fn test_buffer_write_out_of_range() {
        let mut buf = OpenCLBuffer::<f32>::alloc(4).unwrap();
        let err = buf.write(3, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, OpenCLError::BufferOutOfRange { .. }));
    }

    #[test]
    fn test_buffer_read_out_of_range() {
        let buf = OpenCLBuffer::<f32>::alloc(4).unwrap();
        let err = buf.read(2, 5).unwrap_err();
        assert!(matches!(err, OpenCLError::BufferOutOfRange { .. }));
    }

    #[test]
    fn test_buffer_copy() {
        let mut src = OpenCLBuffer::<i32>::alloc(4).unwrap();
        src.write(0, &[10, 20, 30, 40]).unwrap();
        let mut dst = OpenCLBuffer::<i32>::alloc(4).unwrap();
        dst.copy_from(&src, 1, 0, 2).unwrap();
        assert_eq!(dst.read(0, 2).unwrap(), vec![20, 30]);
    }

    #[test]
    fn test_buffer_copy_out_of_range_src() {
        let src = OpenCLBuffer::<i32>::alloc(2).unwrap();
        let mut dst = OpenCLBuffer::<i32>::alloc(4).unwrap();
        let err = dst.copy_from(&src, 1, 0, 3).unwrap_err();
        assert!(matches!(err, OpenCLError::BufferOutOfRange { .. }));
    }

    #[test]
    fn test_buffer_copy_out_of_range_dst() {
        let src = OpenCLBuffer::<i32>::alloc(4).unwrap();
        let mut dst = OpenCLBuffer::<i32>::alloc(2).unwrap();
        let err = dst.copy_from(&src, 0, 1, 3).unwrap_err();
        assert!(matches!(err, OpenCLError::BufferOutOfRange { .. }));
    }

    #[test]
    fn test_buffer_fill() {
        let mut buf = OpenCLBuffer::<f32>::alloc(5).unwrap();
        buf.fill(42.0);
        assert_eq!(buf.read_all(), vec![42.0; 5]);
    }

    #[test]
    fn test_buffer_size_bytes() {
        let buf = OpenCLBuffer::<f32>::alloc(10).unwrap();
        assert_eq!(buf.size_bytes(), 10 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_buffer_read_all_default() {
        let buf = OpenCLBuffer::<u32>::alloc(3).unwrap();
        assert_eq!(buf.read_all(), vec![0, 0, 0]);
    }

    #[test]
    fn test_buffer_overwrite() {
        let mut buf = OpenCLBuffer::<i32>::alloc(4).unwrap();
        buf.write(0, &[1, 2, 3, 4]).unwrap();
        buf.write(0, &[10, 20]).unwrap();
        assert_eq!(buf.read_all(), vec![10, 20, 3, 4]);
    }

    // -- KernelSource tests -----------------------------------------------

    #[test]
    fn test_kernel_source_creation() {
        let ks = OpenCLKernelSource::from_source("__kernel void foo() {}").unwrap();
        assert!(!ks.is_compiled());
        assert!(ks.source().contains("foo"));
    }

    #[test]
    fn test_kernel_source_empty() {
        let err = OpenCLKernelSource::from_source("").unwrap_err();
        assert_eq!(err, OpenCLError::EmptyKernelSource);
    }

    #[test]
    fn test_kernel_source_compile_success() {
        let mut ks = OpenCLKernelSource::from_source("__kernel void add() {}").unwrap();
        ks.compile().unwrap();
        assert!(ks.is_compiled());
        assert!(ks.build_log().unwrap().contains("successful"));
    }

    #[test]
    fn test_kernel_source_compile_error() {
        let mut ks = OpenCLKernelSource::from_source("#error bad").unwrap();
        let err = ks.compile().unwrap_err();
        assert!(matches!(err, OpenCLError::KernelCompilationFailed(_)));
        assert!(!ks.is_compiled());
    }

    #[test]
    fn test_kernel_source_build_options() {
        let mut ks = OpenCLKernelSource::from_source("__kernel void f() {}").unwrap();
        ks.set_build_options("-cl-fast-relaxed-math");
        assert_eq!(ks.build_options(), "-cl-fast-relaxed-math");
    }

    #[test]
    fn test_kernel_source_cached_binary() {
        let mut ks = OpenCLKernelSource::from_source("__kernel void f() {}").unwrap();
        assert!(ks.cached_binary().is_none());
        ks.compile().unwrap();
        assert!(ks.cached_binary().is_some());
    }

    #[test]
    fn test_kernel_source_load_cached() {
        let mut ks = OpenCLKernelSource::from_source("__kernel void f() {}").unwrap();
        ks.load_cached_binary(vec![0xDE, 0xAD]);
        assert!(ks.is_compiled());
        assert_eq!(ks.cached_binary().unwrap(), &[0xDE, 0xAD]);
    }

    #[test]
    fn test_kernel_source_save_to_cache() {
        let mut ks = OpenCLKernelSource::from_source("__kernel void f() {}").unwrap();
        ks.compile().unwrap();
        let path = ks.save_to_cache(&PathBuf::from("/tmp/cache"), "test_kernel").unwrap();
        assert!(path.to_string_lossy().contains("test_kernel.clbin"));
    }

    #[test]
    fn test_kernel_source_save_no_binary() {
        let ks = OpenCLKernelSource::from_source("__kernel void f() {}").unwrap();
        assert!(ks.save_to_cache(&PathBuf::from("/tmp"), "kern").is_none());
    }

    // -- Program tests ----------------------------------------------------

    #[test]
    fn test_program_from_source() {
        let p = OpenCLProgram::from_source("__kernel void add() {}").unwrap();
        assert!(!p.is_built());
    }

    #[test]
    fn test_program_from_empty_source() {
        let err = OpenCLProgram::from_source("").unwrap_err();
        assert_eq!(err, OpenCLError::EmptyKernelSource);
    }

    #[test]
    fn test_program_from_spirv() {
        let p = OpenCLProgram::from_spirv(vec![0x07, 0x23, 0x02, 0x00]).unwrap();
        assert!(!p.is_built());
    }

    #[test]
    fn test_program_from_empty_spirv() {
        let err = OpenCLProgram::from_spirv(vec![]).unwrap_err();
        assert_eq!(err, OpenCLError::EmptyKernelSource);
    }

    #[test]
    fn test_program_build_source() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut p =
            OpenCLProgram::from_source("__kernel void add(int a) {} __kernel void mul(float b) {}")
                .unwrap();
        p.build(&ctx).unwrap();
        assert!(p.is_built());
        assert_eq!(p.kernel_names(), &["add", "mul"]);
    }

    #[test]
    fn test_program_build_spirv() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut p = OpenCLProgram::from_spirv(vec![1, 2, 3]).unwrap();
        p.build(&ctx).unwrap();
        assert!(p.is_built());
        assert_eq!(p.kernel_names(), &["spirv_entry"]);
    }

    #[test]
    fn test_program_build_failure() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut p = OpenCLProgram::from_source("#error fail").unwrap();
        let err = p.build(&ctx).unwrap_err();
        assert!(matches!(err, OpenCLError::KernelCompilationFailed(_)));
        assert!(!p.is_built());
    }

    #[test]
    fn test_program_build_log() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut p = OpenCLProgram::from_source("__kernel void f() {}").unwrap();
        p.build(&ctx).unwrap();
        assert!(p.build_log().unwrap().contains("successful"));
    }

    #[test]
    fn test_program_build_options() {
        let mut p = OpenCLProgram::from_source("__kernel void f() {}").unwrap();
        p.set_build_options("-cl-mad-enable");
        assert_eq!(p.build_options(), "-cl-mad-enable");
    }

    // -- NDRange tests ----------------------------------------------------

    #[test]
    fn test_ndrange_1d() {
        let r = NDRange::new_1d(256);
        assert_eq!(r.dimensions(), 1);
        assert_eq!(r.total_work_items(), 256);
    }

    #[test]
    fn test_ndrange_2d() {
        let r = NDRange::new_2d(64, 32);
        assert_eq!(r.dimensions(), 2);
        assert_eq!(r.total_work_items(), 64 * 32);
    }

    #[test]
    fn test_ndrange_3d() {
        let r = NDRange::new_3d(8, 8, 8);
        assert_eq!(r.dimensions(), 3);
        assert_eq!(r.total_work_items(), 512);
    }

    #[test]
    fn test_ndrange_with_local() {
        let r = NDRange::new_1d(256).with_local(vec![64]).unwrap();
        assert_eq!(r.num_work_groups(), Some(4));
    }

    #[test]
    fn test_ndrange_local_dim_mismatch() {
        let err = NDRange::new_2d(64, 64).with_local(vec![8]).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_ndrange_local_not_divisible() {
        let err = NDRange::new_1d(100).with_local(vec![33]).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_ndrange_local_zero() {
        let err = NDRange::new_1d(64).with_local(vec![0]).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_ndrange_with_offset() {
        let r = NDRange::new_1d(128).with_offset(vec![32]).unwrap();
        assert_eq!(r.offset, Some(vec![32]));
    }

    #[test]
    fn test_ndrange_offset_dim_mismatch() {
        let err = NDRange::new_1d(128).with_offset(vec![0, 0]).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_ndrange_validate_ok() {
        let r = NDRange::new_1d(256).with_local(vec![128]).unwrap();
        r.validate(1024).unwrap();
    }

    #[test]
    fn test_ndrange_validate_exceeds_max_wg() {
        let r = NDRange::new_2d(256, 256).with_local(vec![64, 64]).unwrap();
        let err = r.validate(1024).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_ndrange_validate_zero_global() {
        let r = NDRange { global: vec![0], local: None, offset: None };
        let err = r.validate(1024).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_ndrange_num_work_groups_none() {
        let r = NDRange::new_1d(256);
        assert_eq!(r.num_work_groups(), None);
    }

    // -- KernelLauncher tests ---------------------------------------------

    #[test]
    fn test_launcher_creation() {
        let l = OpenCLKernelLauncher::new("matmul", 4);
        assert_eq!(l.kernel_name(), "matmul");
        assert_eq!(l.arg_count(), 0);
    }

    #[test]
    fn test_launcher_set_arg() {
        let mut l = OpenCLKernelLauncher::new("add", 3);
        l.set_arg(0, KernelArg::Int(42)).unwrap();
        l.set_arg(1, KernelArg::Float(3.14)).unwrap();
        assert_eq!(l.arg_count(), 2);
    }

    #[test]
    fn test_launcher_arg_out_of_range() {
        let mut l = OpenCLKernelLauncher::new("f", 2);
        let err = l.set_arg(5, KernelArg::Int(0)).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidArgIndex { .. }));
    }

    #[test]
    fn test_launcher_get_arg() {
        let mut l = OpenCLKernelLauncher::new("f", 4);
        l.set_arg(2, KernelArg::UInt(99)).unwrap();
        assert!(l.get_arg(2).is_some());
        assert!(l.get_arg(0).is_none());
    }

    #[test]
    fn test_launcher_clear_args() {
        let mut l = OpenCLKernelLauncher::new("f", 4);
        l.set_arg(0, KernelArg::Int(1)).unwrap();
        l.set_arg(1, KernelArg::Int(2)).unwrap();
        l.clear_args();
        assert_eq!(l.arg_count(), 0);
    }

    #[test]
    fn test_launcher_set_ndrange() {
        let mut l = OpenCLKernelLauncher::new("f", 2);
        l.set_nd_range(NDRange::new_1d(256));
        assert!(l.nd_range().is_some());
        assert_eq!(l.nd_range().unwrap().total_work_items(), 256);
    }

    #[test]
    fn test_launcher_enqueue() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut q = OpenCLCommandQueue::new(&ctx, ExecutionMode::InOrder, false);
        let mut l = OpenCLKernelLauncher::new("f", 2);
        l.set_nd_range(NDRange::new_1d(64));
        l.enqueue(&mut q).unwrap();
        assert_eq!(l.launch_count(), 1);
        assert_eq!(q.enqueued_count(), 1);
    }

    #[test]
    fn test_launcher_enqueue_no_range() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut q = OpenCLCommandQueue::new(&ctx, ExecutionMode::InOrder, false);
        let mut l = OpenCLKernelLauncher::new("f", 2);
        let err = l.enqueue(&mut q).unwrap_err();
        assert!(matches!(err, OpenCLError::InvalidNDRange(_)));
    }

    #[test]
    fn test_launcher_multiple_enqueues() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut q = OpenCLCommandQueue::new(&ctx, ExecutionMode::InOrder, false);
        let mut l = OpenCLKernelLauncher::new("f", 2);
        l.set_nd_range(NDRange::new_1d(64));
        for _ in 0..5 {
            l.enqueue(&mut q).unwrap();
        }
        assert_eq!(l.launch_count(), 5);
        assert_eq!(q.enqueued_count(), 5);
    }

    #[test]
    fn test_launcher_buffer_handle_arg() {
        let mut l = OpenCLKernelLauncher::new("f", 3);
        l.set_arg(0, KernelArg::BufferHandle(0x1234)).unwrap();
        assert!(matches!(l.get_arg(0), Some(KernelArg::BufferHandle(0x1234))));
    }

    #[test]
    fn test_launcher_local_memory_arg() {
        let mut l = OpenCLKernelLauncher::new("f", 3);
        l.set_arg(0, KernelArg::LocalMemory(4096)).unwrap();
        assert!(matches!(l.get_arg(0), Some(KernelArg::LocalMemory(4096))));
    }

    // -- Backend tests ----------------------------------------------------

    #[test]
    fn test_backend_new() {
        let b = OpenCLBackend::new(OpenCLConfig::default());
        assert!(!b.is_ready());
        assert!(b.platforms().len() >= 2);
        assert!(b.devices().len() >= 2);
    }

    #[test]
    fn test_backend_initialize() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        assert!(b.is_ready());
        assert!(b.context().is_ok());
    }

    #[test]
    fn test_backend_invalid_platform() {
        let mut b = OpenCLBackend::new(OpenCLConfig { platform_index: 99, ..Default::default() });
        let err = b.initialize().unwrap_err();
        assert!(matches!(err, OpenCLError::PlatformNotFound(99)));
    }

    #[test]
    fn test_backend_invalid_device() {
        let mut b = OpenCLBackend::new(OpenCLConfig { device_index: 99, ..Default::default() });
        let err = b.initialize().unwrap_err();
        assert!(matches!(err, OpenCLError::DeviceNotFound(99)));
    }

    #[test]
    fn test_backend_select_device_by_index() {
        let b = OpenCLBackend::new(OpenCLConfig::default());
        let d = b.select_device(0).unwrap();
        assert_eq!(d.device_type, OpenCLDeviceType::Cpu);
        let d = b.select_device(1).unwrap();
        assert_eq!(d.device_type, OpenCLDeviceType::Gpu);
    }

    #[test]
    fn test_backend_select_device_invalid_index() {
        let b = OpenCLBackend::new(OpenCLConfig::default());
        let err = b.select_device(42).unwrap_err();
        assert!(matches!(err, OpenCLError::DeviceNotFound(42)));
    }

    #[test]
    fn test_backend_select_device_by_type() {
        let b = OpenCLBackend::new(OpenCLConfig::default());
        let (idx, d) = b.select_device_by_type(OpenCLDeviceType::Gpu).unwrap();
        assert_eq!(d.device_type, OpenCLDeviceType::Gpu);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_backend_select_device_by_type_not_found() {
        let b = OpenCLBackend::new(OpenCLConfig::default());
        let err = b.select_device_by_type(OpenCLDeviceType::Accelerator).unwrap_err();
        assert!(matches!(err, OpenCLError::DeviceNotFound(_)));
    }

    #[test]
    fn test_backend_queue_count() {
        let mut b = OpenCLBackend::new(OpenCLConfig { queue_count: 3, ..Default::default() });
        b.initialize().unwrap();
        assert_eq!(b.queue_count(), 3);
    }

    #[test]
    fn test_backend_queue_access() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        assert!(b.queue(0).is_some());
        assert!(b.queue(1).is_none());
    }

    #[test]
    fn test_backend_build_program() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        b.build_program("test", "__kernel void test_k(int a) {}").unwrap();
        let p = b.get_program("test").unwrap();
        assert!(p.is_built());
        assert_eq!(p.kernel_names(), &["test_k"]);
    }

    #[test]
    fn test_backend_build_program_spirv() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        b.build_program_spirv("spv", vec![0x07, 0x23]).unwrap();
        let p = b.get_program("spv").unwrap();
        assert!(p.is_built());
    }

    #[test]
    fn test_backend_build_program_no_context() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        let err = b.build_program("test", "__kernel void f() {}").unwrap_err();
        assert!(matches!(err, OpenCLError::ContextNotInitialized));
    }

    #[test]
    fn test_backend_create_launcher() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        b.build_program("prog", "__kernel void kern(int a) {}").unwrap();
        let l = b.create_launcher("prog", "kern", 4).unwrap();
        assert_eq!(l.kernel_name(), "kern");
    }

    #[test]
    fn test_backend_create_launcher_no_program() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        let err = b.create_launcher("missing", "f", 1).unwrap_err();
        assert!(matches!(err, OpenCLError::ProgramNotBuilt));
    }

    #[test]
    fn test_backend_alloc_buffer() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        let buf = b.alloc_buffer::<f32>(64).unwrap();
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn test_backend_alloc_buffer_not_ready() {
        let b = OpenCLBackend::new(OpenCLConfig::default());
        let err = b.alloc_buffer::<f32>(64).unwrap_err();
        assert!(matches!(err, OpenCLError::BackendNotReady(_)));
    }

    #[test]
    fn test_backend_shutdown() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        assert!(b.is_ready());
        b.shutdown();
        assert!(!b.is_ready());
        assert!(b.context().is_err());
    }

    #[test]
    fn test_backend_profiling_config() {
        let mut b =
            OpenCLBackend::new(OpenCLConfig { profiling_enabled: true, ..Default::default() });
        b.initialize().unwrap();
        let q = b.queue(0).unwrap();
        assert!(q.profiling_enabled());
    }

    #[test]
    fn test_backend_kernel_cache_dir() {
        let cfg = OpenCLConfig {
            kernel_cache_dir: Some(PathBuf::from("/tmp/kernels")),
            ..Default::default()
        };
        let b = OpenCLBackend::new(cfg);
        // Just verify it doesn't panic; real caching is a future feature.
        assert!(!b.is_ready());
    }

    // -- Error display tests ----------------------------------------------

    #[test]
    fn test_error_display_platform_not_found() {
        let e = OpenCLError::PlatformNotFound(5);
        assert!(format!("{e}").contains("5"));
    }

    #[test]
    fn test_error_display_buffer_range() {
        let e = OpenCLError::BufferOutOfRange { offset: 10, len: 20, capacity: 15 };
        let msg = format!("{e}");
        assert!(msg.contains("30"));
        assert!(msg.contains("15"));
    }

    #[test]
    fn test_error_display_invalid_arg() {
        let e = OpenCLError::InvalidArgIndex { index: 5, max: 3 };
        assert!(format!("{e}").contains("5"));
    }

    // -- Config defaults --------------------------------------------------

    #[test]
    fn test_config_default() {
        let c = OpenCLConfig::default();
        assert_eq!(c.platform_index, 0);
        assert_eq!(c.device_index, 0);
        assert!(!c.profiling_enabled);
        assert_eq!(c.queue_count, 1);
        assert!(c.kernel_cache_dir.is_none());
    }

    // -- Integration-style tests ------------------------------------------

    #[test]
    fn test_end_to_end_kernel_execution() {
        let mut backend = OpenCLBackend::new(OpenCLConfig::default());
        backend.initialize().unwrap();

        let source = "__kernel void vector_add(int n) {}";
        backend.build_program("vadd", source).unwrap();

        let mut launcher = backend.create_launcher("vadd", "vector_add", 3).unwrap();
        launcher.set_arg(0, KernelArg::Int(1024)).unwrap();
        launcher.set_nd_range(NDRange::new_1d(1024));

        let q = backend.queue(0).unwrap();
        launcher.enqueue(q).unwrap();

        assert_eq!(launcher.launch_count(), 1);
    }

    #[test]
    fn test_end_to_end_buffer_workflow() {
        let mut backend = OpenCLBackend::new(OpenCLConfig::default());
        backend.initialize().unwrap();

        let mut a = backend.alloc_buffer::<f32>(256).unwrap();
        let mut b = backend.alloc_buffer::<f32>(256).unwrap();

        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        a.write(0, &input).unwrap();
        b.copy_from(&a, 0, 0, 256).unwrap();

        let out = b.read_all();
        assert_eq!(out.len(), 256);
        assert!((out[100] - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multiple_programs() {
        let mut b = OpenCLBackend::new(OpenCLConfig::default());
        b.initialize().unwrap();
        b.build_program("p1", "__kernel void k1() {}").unwrap();
        b.build_program("p2", "__kernel void k2() {}").unwrap();
        assert!(b.get_program("p1").is_some());
        assert!(b.get_program("p2").is_some());
        assert!(b.get_program("p3").is_none());
    }

    #[test]
    fn test_out_of_order_queue_simulation() {
        let ctx = OpenCLContext::new(
            OpenCLPlatformInfo::mock_cpu_platform(),
            OpenCLDeviceInfo::mock_cpu(),
        );
        let mut q = OpenCLCommandQueue::new(&ctx, ExecutionMode::OutOfOrder, false);

        let mut l1 = OpenCLKernelLauncher::new("k1", 1);
        l1.set_nd_range(NDRange::new_1d(64));
        let mut l2 = OpenCLKernelLauncher::new("k2", 1);
        l2.set_nd_range(NDRange::new_1d(128));

        l1.enqueue(&mut q).unwrap();
        l2.enqueue(&mut q).unwrap();
        q.finish();

        assert_eq!(q.enqueued_count(), 2);
    }

    // -- proptest ---------------------------------------------------------

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_buffer_alloc_random_size(size in 1usize..10_000) {
                let buf = OpenCLBuffer::<u8>::alloc(size).unwrap();
                prop_assert_eq!(buf.len(), size);
            }

            #[test]
            fn test_buffer_write_read_roundtrip(
                size in 1usize..1000,
                val in 0i32..1000
            ) {
                let mut buf = OpenCLBuffer::<i32>::alloc(size).unwrap();
                let data = vec![val; size];
                buf.write(0, &data).unwrap();
                let out = buf.read_all();
                prop_assert_eq!(out, data);
            }

            #[test]
            fn test_ndrange_1d_random(global in 1usize..100_000) {
                let r = NDRange::new_1d(global);
                prop_assert_eq!(r.total_work_items(), global);
                prop_assert_eq!(r.dimensions(), 1);
            }

            #[test]
            fn test_ndrange_2d_random(gx in 1usize..1000, gy in 1usize..1000) {
                let r = NDRange::new_2d(gx, gy);
                prop_assert_eq!(r.total_work_items(), gx * gy);
                prop_assert_eq!(r.dimensions(), 2);
            }

            #[test]
            fn test_ndrange_3d_random(
                gx in 1usize..100,
                gy in 1usize..100,
                gz in 1usize..100
            ) {
                let r = NDRange::new_3d(gx, gy, gz);
                prop_assert_eq!(r.total_work_items(), gx * gy * gz);
            }

            #[test]
            fn test_buffer_fill_random(size in 1usize..500, val in 0.0f32..1000.0) {
                let mut buf = OpenCLBuffer::<f32>::alloc(size).unwrap();
                buf.fill(val);
                let data = buf.read_all();
                for v in data {
                    prop_assert!((v - val).abs() < f32::EPSILON);
                }
            }

            #[test]
            fn test_ndrange_local_divides_global(
                factor in 1usize..64,
                multiplier in 1usize..64
            ) {
                let local = factor;
                let global = factor * multiplier;
                let r = NDRange::new_1d(global).with_local(vec![local]).unwrap();
                prop_assert_eq!(r.num_work_groups(), Some(multiplier));
            }

            #[test]
            fn test_buffer_size_bytes_random(size in 1usize..10_000) {
                let buf = OpenCLBuffer::<f64>::alloc(size).unwrap();
                prop_assert_eq!(buf.size_bytes(), size * std::mem::size_of::<f64>());
            }
        }
    }
}
