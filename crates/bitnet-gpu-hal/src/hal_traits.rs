//! Unified GPU HAL traits for backend-agnostic compute.
//!
//! Defines a hardware abstraction layer (HAL) for GPU backends so that
//! higher-level inference code can target CUDA, `OpenCL`, Level Zero, Vulkan
//! Compute, or any future backend through a single set of trait interfaces.

use std::fmt;

// ── Error type ──────────────────────────────────────────────────────────────

/// Unified error type across all GPU HAL backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HalError {
    /// Device was not found or is unavailable.
    DeviceNotFound(String),
    /// Out of device or host memory.
    OutOfMemory { requested: usize, available: usize },
    /// Kernel compilation failed.
    CompilationFailed(String),
    /// Kernel launch or execution error.
    KernelLaunchFailed(String),
    /// Invalid argument index or type.
    InvalidArgument { index: usize, reason: String },
    /// Buffer access error (e.g. out-of-bounds, unmapped).
    BufferAccessError(String),
    /// Queue submission or synchronization failure.
    QueueError(String),
    /// The operation timed out.
    Timeout { operation: String, elapsed_ms: u64 },
    /// Feature not supported by this backend.
    Unsupported(String),
    /// Generic backend-specific error.
    BackendError { backend: String, message: String },
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound(msg) => write!(f, "device not found: {msg}"),
            Self::OutOfMemory { requested, available } => {
                write!(f, "out of memory: requested {requested} B, available {available} B")
            }
            Self::CompilationFailed(msg) => write!(f, "compilation failed: {msg}"),
            Self::KernelLaunchFailed(msg) => write!(f, "kernel launch failed: {msg}"),
            Self::InvalidArgument { index, reason } => {
                write!(f, "invalid argument at index {index}: {reason}")
            }
            Self::BufferAccessError(msg) => write!(f, "buffer access error: {msg}"),
            Self::QueueError(msg) => write!(f, "queue error: {msg}"),
            Self::Timeout { operation, elapsed_ms } => {
                write!(f, "timeout: {operation} after {elapsed_ms} ms")
            }
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::BackendError { backend, message } => {
                write!(f, "[{backend}] {message}")
            }
        }
    }
}

impl std::error::Error for HalError {}

/// Convenience alias.
pub type HalResult<T> = Result<T, HalError>;

// ── Memory type ─────────────────────────────────────────────────────────────

/// Where a buffer is physically allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Device-local (VRAM). Fastest for compute, not host-accessible.
    Device,
    /// Shared / unified memory visible to both host and device.
    Shared,
    /// Host-pinned (page-locked) memory for fast DMA transfers.
    Pinned,
}

// ── Compute capabilities ────────────────────────────────────────────────────

/// Describes the compute capabilities of a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeCapabilities {
    /// Maximum work-group / thread-block size per dimension.
    pub max_workgroup_size: [usize; 3],
    /// Maximum number of work-groups / grid blocks per dimension.
    pub max_grid_size: [usize; 3],
    /// Maximum shared / local memory in bytes.
    pub max_shared_memory_bytes: usize,
    /// Number of compute units (SMs, EUs, CUs, …).
    pub compute_units: usize,
    /// Whether the device supports FP16 compute.
    pub supports_fp16: bool,
    /// Whether the device supports INT8 / DP4A dot-product.
    pub supports_int8: bool,
    /// Whether the device supports sub-group operations.
    pub supports_subgroups: bool,
}

// ── GpuDevice ───────────────────────────────────────────────────────────────

/// A handle to a physical or virtual GPU device.
pub trait GpuDevice: fmt::Debug + Send + Sync {
    /// Human-readable device name (e.g. "Intel Arc A770").
    fn name(&self) -> &str;

    /// Backend-specific vendor identifier.
    fn vendor(&self) -> &str;

    /// Total device memory in bytes.
    fn total_memory(&self) -> usize;

    /// Currently free device memory in bytes.
    fn free_memory(&self) -> usize;

    /// Query compute capabilities of the device.
    fn compute_capabilities(&self) -> ComputeCapabilities;

    /// Whether the device is available and healthy.
    fn is_available(&self) -> bool;
}

// ── GpuBuffer ───────────────────────────────────────────────────────────────

/// A device-side buffer.
pub trait GpuBuffer: fmt::Debug + Send + Sync {
    /// Size of the buffer in bytes.
    fn size(&self) -> usize;

    /// Memory type used for this buffer.
    fn memory_type(&self) -> MemoryType;

    /// Write `data` into the buffer starting at `offset`.
    fn write(&mut self, offset: usize, data: &[u8]) -> HalResult<()>;

    /// Read `len` bytes from the buffer starting at `offset`.
    fn read(&self, offset: usize, len: usize) -> HalResult<Vec<u8>>;

    /// Copy `len` bytes from `src_offset` in this buffer to `dst_offset` in
    /// `dst`.
    fn copy_to(
        &self,
        src_offset: usize,
        dst: &mut dyn GpuBuffer,
        dst_offset: usize,
        len: usize,
    ) -> HalResult<()>;

    /// Map the buffer into host address space, returning a mutable slice.
    /// Only valid for `Shared` or `Pinned` memory types.
    fn map(&mut self) -> HalResult<&mut [u8]>;

    /// Unmap a previously mapped buffer.
    fn unmap(&mut self) -> HalResult<()>;
}

// ── GpuKernel ───────────────────────────────────────────────────────────────

/// A compiled kernel ready for dispatch.
pub trait GpuKernel: fmt::Debug + Send + Sync {
    /// Name of the kernel entry point.
    fn name(&self) -> &str;

    /// Set a scalar/buffer argument at the given index.
    fn set_arg(&mut self, index: usize, data: &[u8]) -> HalResult<()>;

    /// Set local (shared) memory size for the given argument index.
    fn set_local_size(&mut self, index: usize, size: usize) -> HalResult<()>;

    /// Launch the kernel with the given global and local work sizes.
    fn launch(&self, global_work_size: &[usize], local_work_size: &[usize]) -> HalResult<()>;
}

// ── GpuQueue ────────────────────────────────────────────────────────────────

/// A command queue for kernel dispatch and memory operations.
pub trait GpuQueue: fmt::Debug + Send + Sync {
    /// Submit a kernel for execution on this queue.
    fn submit(&mut self, kernel: &dyn GpuKernel) -> HalResult<()>;

    /// Block until all previously submitted work completes.
    fn synchronize(&self) -> HalResult<()>;

    /// Wait for a specific event to complete.
    fn wait_event(&self, event: &dyn GpuEvent) -> HalResult<()>;

    /// Whether profiling is enabled on this queue.
    fn profiling_enabled(&self) -> bool;
}

// ── GpuProgram ──────────────────────────────────────────────────────────────

/// Source language for kernel programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgramSource<'a> {
    /// `OpenCL` C, CUDA C, or similar textual source.
    Source(&'a str),
    /// Pre-compiled SPIR-V binary.
    SpirV(&'a [u8]),
}

/// A compiled program containing one or more kernels.
pub trait GpuProgram: fmt::Debug + Send + Sync {
    /// Compile a program from the given source.
    fn compile(source: ProgramSource<'_>, options: &str) -> HalResult<Self>
    where
        Self: Sized;

    /// Retrieve a kernel by entry-point name.
    fn get_kernel(&self, name: &str) -> HalResult<Box<dyn GpuKernel>>;

    /// List all kernel entry-point names in this program.
    fn kernel_names(&self) -> Vec<String>;
}

// ── GpuEvent ────────────────────────────────────────────────────────────────

/// An event for synchronization and timing.
pub trait GpuEvent: fmt::Debug + Send + Sync {
    /// Block until the event is complete.
    fn wait(&self) -> HalResult<()>;

    /// Whether the event has already completed.
    fn is_complete(&self) -> bool;

    /// Elapsed time in nanoseconds between event start and end.
    /// Returns `None` if profiling data is not available.
    fn elapsed_ns(&self) -> HalResult<Option<u64>>;
}

// ── GpuContext ──────────────────────────────────────────────────────────────

/// A context tying together a device, queue, and compiled programs.
pub trait GpuContext: fmt::Debug + Send + Sync {
    /// The device backing this context.
    fn device(&self) -> &dyn GpuDevice;

    /// The default command queue.
    fn queue(&self) -> &dyn GpuQueue;

    /// A mutable reference to the default command queue.
    fn queue_mut(&mut self) -> &mut dyn GpuQueue;

    /// Allocate a buffer of `size` bytes with the given memory type.
    fn create_buffer(&self, size: usize, memory_type: MemoryType) -> HalResult<Box<dyn GpuBuffer>>;

    /// Compile a program in this context.
    fn compile_program(
        &self,
        source: ProgramSource<'_>,
        options: &str,
    ) -> HalResult<Box<dyn GpuProgram>>;
}

// ── GpuBackend ──────────────────────────────────────────────────────────────

/// A full GPU backend: device enumeration, context creation, program
/// compilation, kernel execution, and cleanup.
pub trait GpuBackend: fmt::Debug + Send + Sync {
    /// Human-readable backend name (e.g. "CUDA", "Level Zero", "`OpenCL`").
    fn backend_name(&self) -> &str;

    /// Enumerate available devices.
    fn enumerate_devices(&self) -> HalResult<Vec<Box<dyn GpuDevice>>>;

    /// Create a context on the given device index.
    fn create_context(&self, device_index: usize) -> HalResult<Box<dyn GpuContext>>;

    /// Execute a compiled kernel with the given arguments on a context.
    fn execute(&self, context: &mut dyn GpuContext, kernel: &dyn GpuKernel) -> HalResult<()>;

    /// Release all resources held by the backend.
    fn cleanup(&mut self) -> HalResult<()>;
}

// ── GpuMemoryAllocator ─────────────────────────────────────────────────────

/// Statistics returned by the memory allocator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocatorStats {
    /// Total bytes currently allocated.
    pub total_allocated: usize,
    /// Number of live allocations.
    pub allocation_count: usize,
    /// Peak bytes ever allocated at one time.
    pub peak_allocated: usize,
}

/// An allocator for device, shared, and pinned memory.
pub trait GpuMemoryAllocator: fmt::Debug + Send + Sync {
    /// Allocate a buffer with the given size and memory type.
    fn alloc(&mut self, size: usize, memory_type: MemoryType) -> HalResult<Box<dyn GpuBuffer>>;

    /// Free a buffer previously allocated by this allocator.
    fn free(&mut self, buffer: Box<dyn GpuBuffer>) -> HalResult<()>;

    /// Attempt to defragment the allocator's memory pools.
    fn defragment(&mut self) -> HalResult<usize>;

    /// Current allocator statistics.
    fn stats(&self) -> AllocatorStats;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests — mock implementations verifying trait contracts
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock device ─────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockDevice {
        name: String,
        vendor: String,
        total_mem: usize,
        free_mem: usize,
        available: bool,
    }

    impl MockDevice {
        fn new() -> Self {
            Self {
                name: "MockGPU 1080Ti".into(),
                vendor: "MockVendor".into(),
                total_mem: 11 * 1024 * 1024 * 1024, // 11 GiB
                free_mem: 8 * 1024 * 1024 * 1024,
                available: true,
            }
        }
    }

    impl GpuDevice for MockDevice {
        fn name(&self) -> &str {
            &self.name
        }
        fn vendor(&self) -> &str {
            &self.vendor
        }
        fn total_memory(&self) -> usize {
            self.total_mem
        }
        fn free_memory(&self) -> usize {
            self.free_mem
        }
        fn compute_capabilities(&self) -> ComputeCapabilities {
            ComputeCapabilities {
                max_workgroup_size: [1024, 1024, 64],
                max_grid_size: [65535, 65535, 65535],
                max_shared_memory_bytes: 49152,
                compute_units: 28,
                supports_fp16: true,
                supports_int8: true,
                supports_subgroups: true,
            }
        }
        fn is_available(&self) -> bool {
            self.available
        }
    }

    // ── Mock buffer ─────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockBuffer {
        data: Vec<u8>,
        mem_type: MemoryType,
        mapped: bool,
    }

    impl MockBuffer {
        fn new(size: usize, mem_type: MemoryType) -> Self {
            Self { data: vec![0u8; size], mem_type, mapped: false }
        }
    }

    impl GpuBuffer for MockBuffer {
        fn size(&self) -> usize {
            self.data.len()
        }
        fn memory_type(&self) -> MemoryType {
            self.mem_type
        }
        fn write(&mut self, offset: usize, data: &[u8]) -> HalResult<()> {
            if offset + data.len() > self.data.len() {
                return Err(HalError::BufferAccessError("write out of bounds".into()));
            }
            self.data[offset..offset + data.len()].copy_from_slice(data);
            Ok(())
        }
        fn read(&self, offset: usize, len: usize) -> HalResult<Vec<u8>> {
            if offset + len > self.data.len() {
                return Err(HalError::BufferAccessError("read out of bounds".into()));
            }
            Ok(self.data[offset..offset + len].to_vec())
        }
        fn copy_to(
            &self,
            src_offset: usize,
            dst: &mut dyn GpuBuffer,
            dst_offset: usize,
            len: usize,
        ) -> HalResult<()> {
            let src_data = self.read(src_offset, len)?;
            dst.write(dst_offset, &src_data)
        }
        fn map(&mut self) -> HalResult<&mut [u8]> {
            if self.mem_type == MemoryType::Device {
                return Err(HalError::BufferAccessError("cannot map device-only buffer".into()));
            }
            self.mapped = true;
            Ok(&mut self.data)
        }
        fn unmap(&mut self) -> HalResult<()> {
            if !self.mapped {
                return Err(HalError::BufferAccessError("buffer is not mapped".into()));
            }
            self.mapped = false;
            Ok(())
        }
    }

    // ── Mock kernel ─────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockKernel {
        name: String,
        args: Vec<Option<Vec<u8>>>,
        local_sizes: Vec<Option<usize>>,
        launched: std::sync::atomic::AtomicBool,
    }

    impl MockKernel {
        fn new(name: &str, num_args: usize) -> Self {
            Self {
                name: name.into(),
                args: vec![None; num_args],
                local_sizes: vec![None; num_args],
                launched: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl GpuKernel for MockKernel {
        fn name(&self) -> &str {
            &self.name
        }
        fn set_arg(&mut self, index: usize, data: &[u8]) -> HalResult<()> {
            if index >= self.args.len() {
                return Err(HalError::InvalidArgument {
                    index,
                    reason: "index out of range".into(),
                });
            }
            self.args[index] = Some(data.to_vec());
            Ok(())
        }
        fn set_local_size(&mut self, index: usize, size: usize) -> HalResult<()> {
            if index >= self.local_sizes.len() {
                return Err(HalError::InvalidArgument {
                    index,
                    reason: "local size index out of range".into(),
                });
            }
            self.local_sizes[index] = Some(size);
            Ok(())
        }
        fn launch(&self, global_work_size: &[usize], local_work_size: &[usize]) -> HalResult<()> {
            if global_work_size.is_empty() || local_work_size.is_empty() {
                return Err(HalError::KernelLaunchFailed("work sizes must be non-empty".into()));
            }
            for (g, l) in global_work_size.iter().zip(local_work_size.iter()) {
                if *l == 0 || g % l != 0 {
                    return Err(HalError::KernelLaunchFailed(format!(
                        "global size {g} not divisible by local size {l}"
                    )));
                }
            }
            self.launched.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    // ── Mock event ──────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockEvent {
        complete: bool,
        elapsed: Option<u64>,
    }

    impl MockEvent {
        fn completed(elapsed_ns: u64) -> Self {
            Self { complete: true, elapsed: Some(elapsed_ns) }
        }
        fn pending() -> Self {
            Self { complete: false, elapsed: None }
        }
    }

    impl GpuEvent for MockEvent {
        fn wait(&self) -> HalResult<()> {
            Ok(())
        }
        fn is_complete(&self) -> bool {
            self.complete
        }
        fn elapsed_ns(&self) -> HalResult<Option<u64>> {
            Ok(self.elapsed)
        }
    }

    // ── Mock queue ──────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockQueue {
        submissions: usize,
        profiling: bool,
    }

    impl MockQueue {
        fn new(profiling: bool) -> Self {
            Self { submissions: 0, profiling }
        }
    }

    impl GpuQueue for MockQueue {
        fn submit(&mut self, _kernel: &dyn GpuKernel) -> HalResult<()> {
            self.submissions += 1;
            Ok(())
        }
        fn synchronize(&self) -> HalResult<()> {
            Ok(())
        }
        fn wait_event(&self, event: &dyn GpuEvent) -> HalResult<()> {
            if event.is_complete() {
                Ok(())
            } else {
                Err(HalError::Timeout { operation: "wait_event".into(), elapsed_ms: 5000 })
            }
        }
        fn profiling_enabled(&self) -> bool {
            self.profiling
        }
    }

    // ── Mock program ────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockProgram {
        kernels: Vec<String>,
    }

    impl GpuProgram for MockProgram {
        fn compile(source: ProgramSource<'_>, _options: &str) -> HalResult<Self> {
            match source {
                ProgramSource::Source("") => {
                    Err(HalError::CompilationFailed("empty source".into()))
                }
                ProgramSource::SpirV([]) => {
                    Err(HalError::CompilationFailed("empty SPIR-V binary".into()))
                }
                _ => Ok(Self { kernels: vec!["mock_kernel".into()] }),
            }
        }
        fn get_kernel(&self, name: &str) -> HalResult<Box<dyn GpuKernel>> {
            if self.kernels.contains(&name.to_string()) {
                Ok(Box::new(MockKernel::new(name, 4)))
            } else {
                Err(HalError::InvalidArgument {
                    index: 0,
                    reason: format!("kernel '{name}' not found"),
                })
            }
        }
        fn kernel_names(&self) -> Vec<String> {
            self.kernels.clone()
        }
    }

    // ── Mock context ────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockContext {
        device: MockDevice,
        queue: MockQueue,
    }

    impl MockContext {
        fn new() -> Self {
            Self { device: MockDevice::new(), queue: MockQueue::new(true) }
        }
    }

    impl GpuContext for MockContext {
        fn device(&self) -> &dyn GpuDevice {
            &self.device
        }
        fn queue(&self) -> &dyn GpuQueue {
            &self.queue
        }
        fn queue_mut(&mut self) -> &mut dyn GpuQueue {
            &mut self.queue
        }
        fn create_buffer(
            &self,
            size: usize,
            memory_type: MemoryType,
        ) -> HalResult<Box<dyn GpuBuffer>> {
            if size > self.device.free_memory() {
                return Err(HalError::OutOfMemory {
                    requested: size,
                    available: self.device.free_memory(),
                });
            }
            Ok(Box::new(MockBuffer::new(size, memory_type)))
        }
        fn compile_program(
            &self,
            source: ProgramSource<'_>,
            options: &str,
        ) -> HalResult<Box<dyn GpuProgram>> {
            let prog = MockProgram::compile(source, options)?;
            Ok(Box::new(prog))
        }
    }

    // ── Mock backend ────────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockBackend {
        name: String,
        devices: Vec<MockDevice>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self { name: "MockBackend".into(), devices: vec![MockDevice::new()] }
        }
    }

    impl GpuBackend for MockBackend {
        fn backend_name(&self) -> &str {
            &self.name
        }
        fn enumerate_devices(&self) -> HalResult<Vec<Box<dyn GpuDevice>>> {
            Ok(self
                .devices
                .iter()
                .map(|d| {
                    Box::new(MockDevice {
                        name: d.name.clone(),
                        vendor: d.vendor.clone(),
                        total_mem: d.total_mem,
                        free_mem: d.free_mem,
                        available: d.available,
                    }) as Box<dyn GpuDevice>
                })
                .collect())
        }
        fn create_context(&self, device_index: usize) -> HalResult<Box<dyn GpuContext>> {
            if device_index >= self.devices.len() {
                return Err(HalError::DeviceNotFound(format!(
                    "device index {device_index} out of range"
                )));
            }
            Ok(Box::new(MockContext::new()))
        }
        fn execute(&self, context: &mut dyn GpuContext, kernel: &dyn GpuKernel) -> HalResult<()> {
            context.queue_mut().submit(kernel)
        }
        fn cleanup(&mut self) -> HalResult<()> {
            self.devices.clear();
            Ok(())
        }
    }

    // ── Mock allocator ──────────────────────────────────────────────────

    #[derive(Debug)]
    struct MockAllocator {
        allocated: usize,
        count: usize,
        peak: usize,
        max_memory: usize,
    }

    impl MockAllocator {
        fn new(max_memory: usize) -> Self {
            Self { allocated: 0, count: 0, peak: 0, max_memory }
        }
    }

    impl GpuMemoryAllocator for MockAllocator {
        fn alloc(&mut self, size: usize, memory_type: MemoryType) -> HalResult<Box<dyn GpuBuffer>> {
            if self.allocated + size > self.max_memory {
                return Err(HalError::OutOfMemory {
                    requested: size,
                    available: self.max_memory - self.allocated,
                });
            }
            self.allocated += size;
            self.count += 1;
            if self.allocated > self.peak {
                self.peak = self.allocated;
            }
            Ok(Box::new(MockBuffer::new(size, memory_type)))
        }
        fn free(&mut self, buffer: Box<dyn GpuBuffer>) -> HalResult<()> {
            let size = buffer.size();
            if size > self.allocated {
                return Err(HalError::BufferAccessError(
                    "freeing more memory than allocated".into(),
                ));
            }
            self.allocated -= size;
            self.count -= 1;
            Ok(())
        }
        fn defragment(&mut self) -> HalResult<usize> {
            // Mock: reclaim 10% of allocated as "freed fragments"
            let reclaimed = self.allocated / 10;
            Ok(reclaimed)
        }
        fn stats(&self) -> AllocatorStats {
            AllocatorStats {
                total_allocated: self.allocated,
                allocation_count: self.count,
                peak_allocated: self.peak,
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // HalError tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn error_display_device_not_found() {
        let e = HalError::DeviceNotFound("gpu0".into());
        assert_eq!(e.to_string(), "device not found: gpu0");
    }

    #[test]
    fn error_display_out_of_memory() {
        let e = HalError::OutOfMemory { requested: 1024, available: 512 };
        assert!(e.to_string().contains("1024"));
        assert!(e.to_string().contains("512"));
    }

    #[test]
    fn error_display_compilation_failed() {
        let e = HalError::CompilationFailed("syntax error".into());
        assert!(e.to_string().contains("syntax error"));
    }

    #[test]
    fn error_display_kernel_launch() {
        let e = HalError::KernelLaunchFailed("bad dims".into());
        assert!(e.to_string().contains("bad dims"));
    }

    #[test]
    fn error_display_invalid_argument() {
        let e = HalError::InvalidArgument { index: 3, reason: "type mismatch".into() };
        let s = e.to_string();
        assert!(s.contains('3') && s.contains("type mismatch"));
    }

    #[test]
    fn error_display_buffer_access() {
        let e = HalError::BufferAccessError("oob".into());
        assert!(e.to_string().contains("oob"));
    }

    #[test]
    fn error_display_queue_error() {
        let e = HalError::QueueError("stall".into());
        assert!(e.to_string().contains("stall"));
    }

    #[test]
    fn error_display_timeout() {
        let e = HalError::Timeout { operation: "sync".into(), elapsed_ms: 3000 };
        let s = e.to_string();
        assert!(s.contains("sync") && s.contains("3000"));
    }

    #[test]
    fn error_display_unsupported() {
        let e = HalError::Unsupported("fp64".into());
        assert!(e.to_string().contains("fp64"));
    }

    #[test]
    fn error_display_backend_error() {
        let e =
            HalError::BackendError { backend: "CUDA".into(), message: "driver mismatch".into() };
        let s = e.to_string();
        assert!(s.contains("CUDA") && s.contains("driver mismatch"));
    }

    #[test]
    fn error_equality() {
        let a = HalError::DeviceNotFound("x".into());
        let b = HalError::DeviceNotFound("x".into());
        assert_eq!(a, b);
    }

    #[test]
    fn error_inequality() {
        let a = HalError::DeviceNotFound("x".into());
        let b = HalError::DeviceNotFound("y".into());
        assert_ne!(a, b);
    }

    #[test]
    fn error_clone() {
        let e = HalError::OutOfMemory { requested: 100, available: 50 };
        let e2 = e.clone();
        assert_eq!(e, e2);
    }

    #[test]
    fn error_debug_format() {
        let e = HalError::Unsupported("test".into());
        let dbg = format!("{e:?}");
        assert!(dbg.contains("Unsupported"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(HalError::QueueError("q".into()));
        assert!(e.to_string().contains("queue error"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // MemoryType tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn memory_type_equality() {
        assert_eq!(MemoryType::Device, MemoryType::Device);
        assert_ne!(MemoryType::Device, MemoryType::Shared);
        assert_ne!(MemoryType::Shared, MemoryType::Pinned);
    }

    #[test]
    fn memory_type_clone() {
        let m = MemoryType::Pinned;
        assert_eq!(m, m.clone());
    }

    #[test]
    fn memory_type_debug() {
        let s = format!("{:?}", MemoryType::Shared);
        assert_eq!(s, "Shared");
    }

    #[test]
    fn memory_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MemoryType::Device);
        set.insert(MemoryType::Shared);
        set.insert(MemoryType::Pinned);
        assert_eq!(set.len(), 3);
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuDevice tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn device_name() {
        let d = MockDevice::new();
        assert_eq!(d.name(), "MockGPU 1080Ti");
    }

    #[test]
    fn device_vendor() {
        let d = MockDevice::new();
        assert_eq!(d.vendor(), "MockVendor");
    }

    #[test]
    fn device_total_memory() {
        let d = MockDevice::new();
        assert_eq!(d.total_memory(), 11 * 1024 * 1024 * 1024);
    }

    #[test]
    fn device_free_memory_less_than_total() {
        let d = MockDevice::new();
        assert!(d.free_memory() <= d.total_memory());
    }

    #[test]
    fn device_available() {
        let d = MockDevice::new();
        assert!(d.is_available());
    }

    #[test]
    fn device_unavailable() {
        let d = MockDevice { available: false, ..MockDevice::new() };
        assert!(!d.is_available());
    }

    #[test]
    fn device_compute_capabilities() {
        let d = MockDevice::new();
        let caps = d.compute_capabilities();
        assert_eq!(caps.compute_units, 28);
        assert!(caps.supports_fp16);
        assert!(caps.supports_int8);
        assert!(caps.supports_subgroups);
    }

    #[test]
    fn device_workgroup_size() {
        let d = MockDevice::new();
        let caps = d.compute_capabilities();
        assert_eq!(caps.max_workgroup_size, [1024, 1024, 64]);
    }

    #[test]
    fn device_grid_size() {
        let d = MockDevice::new();
        let caps = d.compute_capabilities();
        assert_eq!(caps.max_grid_size, [65535, 65535, 65535]);
    }

    #[test]
    fn device_shared_memory() {
        let d = MockDevice::new();
        let caps = d.compute_capabilities();
        assert_eq!(caps.max_shared_memory_bytes, 49152);
    }

    #[test]
    fn device_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockDevice>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuBuffer tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn buffer_size() {
        let buf = MockBuffer::new(256, MemoryType::Device);
        assert_eq!(buf.size(), 256);
    }

    #[test]
    fn buffer_memory_type() {
        let buf = MockBuffer::new(64, MemoryType::Shared);
        assert_eq!(buf.memory_type(), MemoryType::Shared);
    }

    #[test]
    fn buffer_write_read_roundtrip() {
        let mut buf = MockBuffer::new(16, MemoryType::Device);
        buf.write(0, &[1, 2, 3, 4]).unwrap();
        let data = buf.read(0, 4).unwrap();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn buffer_write_at_offset() {
        let mut buf = MockBuffer::new(16, MemoryType::Device);
        buf.write(8, &[0xAA, 0xBB]).unwrap();
        let data = buf.read(8, 2).unwrap();
        assert_eq!(data, vec![0xAA, 0xBB]);
    }

    #[test]
    fn buffer_write_out_of_bounds() {
        let mut buf = MockBuffer::new(4, MemoryType::Device);
        let result = buf.write(2, &[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn buffer_read_out_of_bounds() {
        let buf = MockBuffer::new(4, MemoryType::Device);
        let result = buf.read(2, 4);
        assert!(result.is_err());
    }

    #[test]
    fn buffer_copy_to() {
        let mut src = MockBuffer::new(8, MemoryType::Device);
        src.write(0, &[10, 20, 30, 40]).unwrap();
        let mut dst = MockBuffer::new(8, MemoryType::Device);
        src.copy_to(0, &mut dst, 4, 4).unwrap();
        let data = dst.read(4, 4).unwrap();
        assert_eq!(data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn buffer_map_shared() {
        let mut buf = MockBuffer::new(8, MemoryType::Shared);
        let mapped = buf.map().unwrap();
        mapped[0] = 42;
        assert_eq!(mapped[0], 42);
    }

    #[test]
    fn buffer_map_pinned() {
        let mut buf = MockBuffer::new(8, MemoryType::Pinned);
        assert!(buf.map().is_ok());
    }

    #[test]
    fn buffer_map_device_fails() {
        let mut buf = MockBuffer::new(8, MemoryType::Device);
        assert!(buf.map().is_err());
    }

    #[test]
    fn buffer_unmap_without_map_fails() {
        let mut buf = MockBuffer::new(8, MemoryType::Shared);
        assert!(buf.unmap().is_err());
    }

    #[test]
    fn buffer_map_unmap_cycle() {
        let mut buf = MockBuffer::new(8, MemoryType::Shared);
        buf.map().unwrap();
        buf.unmap().unwrap();
    }

    #[test]
    fn buffer_initial_zeroed() {
        let buf = MockBuffer::new(4, MemoryType::Device);
        assert_eq!(buf.read(0, 4).unwrap(), vec![0, 0, 0, 0]);
    }

    #[test]
    fn buffer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockBuffer>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuKernel tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn kernel_name() {
        let k = MockKernel::new("vec_add", 3);
        assert_eq!(k.name(), "vec_add");
    }

    #[test]
    fn kernel_set_arg() {
        let mut k = MockKernel::new("k", 2);
        k.set_arg(0, &[1, 2, 3, 4]).unwrap();
        assert!(k.args[0].is_some());
    }

    #[test]
    fn kernel_set_arg_out_of_range() {
        let mut k = MockKernel::new("k", 2);
        assert!(k.set_arg(5, &[1]).is_err());
    }

    #[test]
    fn kernel_set_local_size() {
        let mut k = MockKernel::new("k", 2);
        k.set_local_size(0, 256).unwrap();
        assert_eq!(k.local_sizes[0], Some(256));
    }

    #[test]
    fn kernel_set_local_size_out_of_range() {
        let mut k = MockKernel::new("k", 1);
        assert!(k.set_local_size(5, 256).is_err());
    }

    #[test]
    fn kernel_launch_success() {
        let k = MockKernel::new("k", 2);
        k.launch(&[256], &[64]).unwrap();
        assert!(k.launched.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn kernel_launch_empty_work_size() {
        let k = MockKernel::new("k", 0);
        assert!(k.launch(&[], &[64]).is_err());
    }

    #[test]
    fn kernel_launch_indivisible_work_size() {
        let k = MockKernel::new("k", 0);
        assert!(k.launch(&[100], &[64]).is_err());
    }

    #[test]
    fn kernel_launch_2d() {
        let k = MockKernel::new("k", 0);
        k.launch(&[256, 128], &[16, 16]).unwrap();
    }

    #[test]
    fn kernel_launch_3d() {
        let k = MockKernel::new("k", 0);
        k.launch(&[64, 64, 64], &[8, 8, 8]).unwrap();
    }

    #[test]
    fn kernel_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockKernel>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuEvent tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn event_completed() {
        let e = MockEvent::completed(1_000_000);
        assert!(e.is_complete());
    }

    #[test]
    fn event_pending() {
        let e = MockEvent::pending();
        assert!(!e.is_complete());
    }

    #[test]
    fn event_elapsed_ns() {
        let e = MockEvent::completed(42_000);
        assert_eq!(e.elapsed_ns().unwrap(), Some(42_000));
    }

    #[test]
    fn event_pending_no_elapsed() {
        let e = MockEvent::pending();
        assert_eq!(e.elapsed_ns().unwrap(), None);
    }

    #[test]
    fn event_wait() {
        let e = MockEvent::completed(100);
        e.wait().unwrap();
    }

    #[test]
    fn event_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockEvent>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuQueue tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn queue_submit() {
        let mut q = MockQueue::new(false);
        let k = MockKernel::new("k", 0);
        q.submit(&k).unwrap();
        assert_eq!(q.submissions, 1);
    }

    #[test]
    fn queue_multiple_submits() {
        let mut q = MockQueue::new(false);
        let k = MockKernel::new("k", 0);
        q.submit(&k).unwrap();
        q.submit(&k).unwrap();
        q.submit(&k).unwrap();
        assert_eq!(q.submissions, 3);
    }

    #[test]
    fn queue_synchronize() {
        let q = MockQueue::new(false);
        q.synchronize().unwrap();
    }

    #[test]
    fn queue_wait_completed_event() {
        let q = MockQueue::new(false);
        let e = MockEvent::completed(100);
        q.wait_event(&e).unwrap();
    }

    #[test]
    fn queue_wait_pending_event_timeout() {
        let q = MockQueue::new(false);
        let e = MockEvent::pending();
        assert!(q.wait_event(&e).is_err());
    }

    #[test]
    fn queue_profiling_enabled() {
        let q = MockQueue::new(true);
        assert!(q.profiling_enabled());
    }

    #[test]
    fn queue_profiling_disabled() {
        let q = MockQueue::new(false);
        assert!(!q.profiling_enabled());
    }

    #[test]
    fn queue_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockQueue>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuProgram tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn program_compile_source() {
        let p = MockProgram::compile(ProgramSource::Source("kernel void k(){}"), "").unwrap();
        assert!(!p.kernel_names().is_empty());
    }

    #[test]
    fn program_compile_spirv() {
        let p = MockProgram::compile(ProgramSource::SpirV(&[0x03, 0x02, 0x23]), "").unwrap();
        assert!(!p.kernel_names().is_empty());
    }

    #[test]
    fn program_compile_empty_source_fails() {
        let r = MockProgram::compile(ProgramSource::Source(""), "");
        assert!(r.is_err());
    }

    #[test]
    fn program_compile_empty_spirv_fails() {
        let r = MockProgram::compile(ProgramSource::SpirV(&[]), "");
        assert!(r.is_err());
    }

    #[test]
    fn program_get_kernel() {
        let p = MockProgram::compile(ProgramSource::Source("code"), "").unwrap();
        let k = p.get_kernel("mock_kernel").unwrap();
        assert_eq!(k.name(), "mock_kernel");
    }

    #[test]
    fn program_get_missing_kernel() {
        let p = MockProgram::compile(ProgramSource::Source("code"), "").unwrap();
        assert!(p.get_kernel("nonexistent").is_err());
    }

    #[test]
    fn program_kernel_names() {
        let p = MockProgram::compile(ProgramSource::Source("code"), "").unwrap();
        let names = p.kernel_names();
        assert!(names.contains(&"mock_kernel".to_string()));
    }

    #[test]
    fn program_source_enum_equality() {
        let a = ProgramSource::Source("hello");
        let b = ProgramSource::Source("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn program_source_enum_inequality() {
        let a = ProgramSource::Source("hello");
        let b = ProgramSource::SpirV(b"hello");
        assert_ne!(a, b);
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuContext tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn context_device_name() {
        let ctx = MockContext::new();
        assert_eq!(ctx.device().name(), "MockGPU 1080Ti");
    }

    #[test]
    fn context_queue_profiling() {
        let ctx = MockContext::new();
        assert!(ctx.queue().profiling_enabled());
    }

    #[test]
    fn context_create_device_buffer() {
        let ctx = MockContext::new();
        let buf = ctx.create_buffer(1024, MemoryType::Device).unwrap();
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.memory_type(), MemoryType::Device);
    }

    #[test]
    fn context_create_shared_buffer() {
        let ctx = MockContext::new();
        let buf = ctx.create_buffer(512, MemoryType::Shared).unwrap();
        assert_eq!(buf.memory_type(), MemoryType::Shared);
    }

    #[test]
    fn context_create_buffer_oom() {
        let ctx = MockContext::new();
        let result = ctx.create_buffer(usize::MAX, MemoryType::Device);
        assert!(result.is_err());
    }

    #[test]
    fn context_compile_program() {
        let ctx = MockContext::new();
        let prog = ctx.compile_program(ProgramSource::Source("code"), "").unwrap();
        assert!(!prog.kernel_names().is_empty());
    }

    #[test]
    fn context_compile_empty_fails() {
        let ctx = MockContext::new();
        assert!(ctx.compile_program(ProgramSource::Source(""), "").is_err());
    }

    #[test]
    fn context_queue_mut_submit() {
        let mut ctx = MockContext::new();
        let k = MockKernel::new("k", 0);
        ctx.queue_mut().submit(&k).unwrap();
    }

    #[test]
    fn context_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockContext>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuBackend tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn backend_name() {
        let b = MockBackend::new();
        assert_eq!(b.backend_name(), "MockBackend");
    }

    #[test]
    fn backend_enumerate_devices() {
        let b = MockBackend::new();
        let devs = b.enumerate_devices().unwrap();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].name(), "MockGPU 1080Ti");
    }

    #[test]
    fn backend_create_context() {
        let b = MockBackend::new();
        let ctx = b.create_context(0).unwrap();
        assert!(ctx.device().is_available());
    }

    #[test]
    fn backend_create_context_invalid_index() {
        let b = MockBackend::new();
        assert!(b.create_context(99).is_err());
    }

    #[test]
    fn backend_execute() {
        let b = MockBackend::new();
        let mut ctx = b.create_context(0).unwrap();
        let k = MockKernel::new("k", 0);
        b.execute(ctx.as_mut(), &k).unwrap();
    }

    #[test]
    fn backend_cleanup() {
        let mut b = MockBackend::new();
        b.cleanup().unwrap();
        assert!(b.devices.is_empty());
    }

    #[test]
    fn backend_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockBackend>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // GpuMemoryAllocator tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn allocator_alloc() {
        let mut a = MockAllocator::new(4096);
        let buf = a.alloc(256, MemoryType::Device).unwrap();
        assert_eq!(buf.size(), 256);
    }

    #[test]
    fn allocator_stats_after_alloc() {
        let mut a = MockAllocator::new(4096);
        a.alloc(100, MemoryType::Device).unwrap();
        let s = a.stats();
        assert_eq!(s.total_allocated, 100);
        assert_eq!(s.allocation_count, 1);
        assert_eq!(s.peak_allocated, 100);
    }

    #[test]
    fn allocator_free() {
        let mut a = MockAllocator::new(4096);
        let buf = a.alloc(100, MemoryType::Device).unwrap();
        a.free(buf).unwrap();
        let s = a.stats();
        assert_eq!(s.total_allocated, 0);
        assert_eq!(s.allocation_count, 0);
    }

    #[test]
    fn allocator_peak_tracks_maximum() {
        let mut a = MockAllocator::new(4096);
        let b1 = a.alloc(200, MemoryType::Device).unwrap();
        let _b2 = a.alloc(300, MemoryType::Shared).unwrap();
        assert_eq!(a.stats().peak_allocated, 500);
        a.free(b1).unwrap();
        // Peak should remain at 500 even after free.
        assert_eq!(a.stats().peak_allocated, 500);
    }

    #[test]
    fn allocator_oom() {
        let mut a = MockAllocator::new(100);
        assert!(a.alloc(200, MemoryType::Device).is_err());
    }

    #[test]
    fn allocator_oom_preserves_state() {
        let mut a = MockAllocator::new(100);
        a.alloc(80, MemoryType::Device).unwrap();
        assert!(a.alloc(50, MemoryType::Device).is_err());
        assert_eq!(a.stats().total_allocated, 80);
    }

    #[test]
    fn allocator_defragment() {
        let mut a = MockAllocator::new(4096);
        a.alloc(1000, MemoryType::Device).unwrap();
        let reclaimed = a.defragment().unwrap();
        assert_eq!(reclaimed, 100); // 10% of 1000
    }

    #[test]
    fn allocator_defragment_empty() {
        let mut a = MockAllocator::new(4096);
        let reclaimed = a.defragment().unwrap();
        assert_eq!(reclaimed, 0);
    }

    #[test]
    fn allocator_multiple_types() {
        let mut a = MockAllocator::new(4096);
        let b1 = a.alloc(100, MemoryType::Device).unwrap();
        let b2 = a.alloc(100, MemoryType::Shared).unwrap();
        let b3 = a.alloc(100, MemoryType::Pinned).unwrap();
        assert_eq!(b1.memory_type(), MemoryType::Device);
        assert_eq!(b2.memory_type(), MemoryType::Shared);
        assert_eq!(b3.memory_type(), MemoryType::Pinned);
        assert_eq!(a.stats().allocation_count, 3);
    }

    #[test]
    fn allocator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockAllocator>();
    }

    // ═════════════════════════════════════════════════════════════════════
    // AllocatorStats tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn allocator_stats_equality() {
        let a = AllocatorStats { total_allocated: 100, allocation_count: 2, peak_allocated: 200 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn allocator_stats_debug() {
        let a = AllocatorStats { total_allocated: 0, allocation_count: 0, peak_allocated: 0 };
        let s = format!("{a:?}");
        assert!(s.contains("AllocatorStats"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // ComputeCapabilities tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn compute_caps_clone_eq() {
        let c = ComputeCapabilities {
            max_workgroup_size: [256, 256, 64],
            max_grid_size: [65535, 65535, 65535],
            max_shared_memory_bytes: 49152,
            compute_units: 80,
            supports_fp16: true,
            supports_int8: false,
            supports_subgroups: true,
        };
        assert_eq!(c.clone(), c);
    }

    #[test]
    fn compute_caps_debug() {
        let c = ComputeCapabilities {
            max_workgroup_size: [1, 1, 1],
            max_grid_size: [1, 1, 1],
            max_shared_memory_bytes: 0,
            compute_units: 1,
            supports_fp16: false,
            supports_int8: false,
            supports_subgroups: false,
        };
        let s = format!("{c:?}");
        assert!(s.contains("ComputeCapabilities"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Integration / workflow tests
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn end_to_end_kernel_workflow() {
        let backend = MockBackend::new();
        let mut ctx = backend.create_context(0).unwrap();

        // Compile program
        let prog = ctx.compile_program(ProgramSource::Source("__kernel void add(){}"), "").unwrap();

        // Get kernel, set args
        let mut kernel = prog.get_kernel("mock_kernel").unwrap();
        kernel.set_arg(0, &[1, 0, 0, 0]).unwrap();
        kernel.set_arg(1, &[2, 0, 0, 0]).unwrap();

        // Create buffers
        let mut buf = ctx.create_buffer(1024, MemoryType::Device).unwrap();
        buf.write(0, &[0xFF; 64]).unwrap();

        // Submit
        ctx.queue_mut().submit(kernel.as_ref()).unwrap();
        ctx.queue().synchronize().unwrap();
    }

    #[test]
    fn end_to_end_allocator_workflow() {
        let mut alloc = MockAllocator::new(8192);
        let b1 = alloc.alloc(1024, MemoryType::Device).unwrap();
        let b2 = alloc.alloc(2048, MemoryType::Shared).unwrap();

        assert_eq!(alloc.stats().total_allocated, 3072);
        assert_eq!(alloc.stats().allocation_count, 2);

        alloc.free(b1).unwrap();
        assert_eq!(alloc.stats().total_allocated, 2048);

        let reclaimed = alloc.defragment().unwrap();
        assert!(reclaimed <= alloc.stats().total_allocated);

        alloc.free(b2).unwrap();
        assert_eq!(alloc.stats().total_allocated, 0);
    }

    #[test]
    fn end_to_end_buffer_transfer() {
        let ctx = MockContext::new();
        let mut src = ctx.create_buffer(256, MemoryType::Device).unwrap();
        let mut dst = ctx.create_buffer(256, MemoryType::Device).unwrap();

        let pattern: Vec<u8> = (0..128).collect();
        src.write(0, &pattern).unwrap();
        src.copy_to(0, dst.as_mut(), 64, 128).unwrap();

        let result = dst.read(64, 128).unwrap();
        assert_eq!(result, pattern);
    }

    #[test]
    fn end_to_end_map_write_unmap() {
        let ctx = MockContext::new();
        let mut buf = ctx.create_buffer(64, MemoryType::Shared).unwrap();
        {
            let mapped = buf.map().unwrap();
            for (i, byte) in mapped.iter_mut().enumerate() {
                *byte = u8::try_from(i).unwrap();
            }
        }
        buf.unmap().unwrap();

        let data = buf.read(0, 64).unwrap();
        for (i, &b) in data.iter().enumerate() {
            assert_eq!(b, u8::try_from(i).unwrap());
        }
    }

    #[test]
    fn backend_cleanup_then_no_devices() {
        let mut b = MockBackend::new();
        assert_eq!(b.enumerate_devices().unwrap().len(), 1);
        b.cleanup().unwrap();
        assert_eq!(b.enumerate_devices().unwrap().len(), 0);
    }
}
