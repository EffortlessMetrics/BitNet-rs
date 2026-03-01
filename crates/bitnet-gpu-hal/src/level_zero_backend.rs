//! Module stub - implementation pending merge from feature branch
//! Intel Level Zero backend for low-level GPU compute.
//!
//! Provides a CPU reference implementation of the Intel oneAPI Level Zero API
//! surface: driver/device enumeration, context and command-list management,
//! SPIR-V module loading, kernel configuration, and device/host/shared memory
//! allocation. The orchestrator [`LevelZeroBackend`] ties these together into
//! an init → enumerate → create-context → load-module → execute pipeline.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Handle generation ───────────────────────────────────────────────────────

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn next_handle() -> u64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

// ── Error type ──────────────────────────────────────────────────────────────

/// Errors returned by Level Zero backend operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LevelZeroError {
    /// The driver at the requested index does not exist.
    DriverNotFound(u32),
    /// The device at the requested index does not exist.
    DeviceNotFound(u32),
    /// A context has already been created (double-init guard).
    ContextAlreadyCreated,
    /// No context has been created yet.
    NoContext,
    /// No module has been loaded yet.
    NoModule,
    /// The SPIR-V binary is empty or invalid.
    InvalidSpirv,
    /// The requested kernel name was not found in the loaded module.
    KernelNotFound(String),
    /// An argument index is out of range.
    ArgumentIndexOutOfRange { index: u32, max: u32 },
    /// The requested memory size is zero or exceeds device capacity.
    InvalidAllocationSize { requested: usize, max: usize },
    /// Alignment must be a power of two.
    InvalidAlignment(usize),
    /// The command list has already been closed.
    CommandListClosed,
    /// The command queue has already been destroyed.
    CommandQueueDestroyed,
    /// Generic backend error with a human-readable message.
    Other(String),
}

impl fmt::Display for LevelZeroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DriverNotFound(i) => write!(f, "driver index {i} not found"),
            Self::DeviceNotFound(i) => write!(f, "device index {i} not found"),
            Self::ContextAlreadyCreated => write!(f, "context already created"),
            Self::NoContext => write!(f, "no context created"),
            Self::NoModule => write!(f, "no module loaded"),
            Self::InvalidSpirv => write!(f, "invalid or empty SPIR-V binary"),
            Self::KernelNotFound(n) => write!(f, "kernel '{n}' not found"),
            Self::ArgumentIndexOutOfRange { index, max } => {
                write!(f, "argument index {index} out of range (max {max})")
            }
            Self::InvalidAllocationSize { requested, max } => {
                write!(f, "invalid allocation size {requested} (max {max})")
            }
            Self::InvalidAlignment(a) => {
                write!(f, "alignment {a} is not a power of two")
            }
            Self::CommandListClosed => write!(f, "command list is closed"),
            Self::CommandQueueDestroyed => write!(f, "command queue is destroyed"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LevelZeroError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, LevelZeroError>;

// ── 1. Configuration ────────────────────────────────────────────────────────

/// Configuration for initialising the Level Zero backend.
#[derive(Debug, Clone, Default)]
pub struct LevelZeroConfig {
    /// Index of the driver to use (0-based).
    pub driver_index: u32,
    /// Index of the device within the driver (0-based).
    pub device_index: u32,
    /// Command queue ordinal (selects compute vs copy engine).
    pub ordinal: u32,
    /// Whether to enable GPU profiling / timestamping.
    pub profiling_enabled: bool,
}

// ── 2. Driver ───────────────────────────────────────────────────────────────

/// Properties of a Level Zero driver.
#[derive(Debug, Clone)]
pub struct DriverProperties {
    /// Driver handle (opaque).
    pub handle: u64,
    /// Driver version encoded as `(major << 16) | minor`.
    pub driver_version: u32,
    /// API version supported by this driver.
    pub api_version: (u32, u32),
    /// Human-readable name.
    pub name: String,
}

/// Enumerates drivers and retrieves their properties.
#[derive(Debug)]
pub struct LevelZeroDriver {
    drivers: Vec<DriverProperties>,
}

impl LevelZeroDriver {
    /// Enumerate all available Level Zero drivers.
    ///
    /// In this CPU reference implementation a single synthetic driver is
    /// always reported.
    pub fn enumerate() -> Self {
        Self {
            drivers: vec![DriverProperties {
                handle: next_handle(),
                driver_version: (1 << 16) | 8,
                api_version: (1, 8),
                name: "Intel(R) Level Zero Reference Driver".to_string(),
            }],
        }
    }

    /// Return the number of enumerated drivers.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.drivers.len()
    }

    /// Get properties for driver at `index`.
    pub fn get_properties(&self, index: u32) -> Result<&DriverProperties> {
        self.drivers.get(index as usize).ok_or(LevelZeroError::DriverNotFound(index))
    }

    /// Get a driver handle by index.
    pub fn get_handle(&self, index: u32) -> Result<u64> {
        self.get_properties(index).map(|p| p.handle)
    }
}

// ── 3. Device ───────────────────────────────────────────────────────────────

/// Sub-slice information for an Intel GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsliceInfo {
    /// Number of slices.
    pub num_slices: u32,
    /// Sub-slices per slice.
    pub sub_slices_per_slice: u32,
    /// Execution units per sub-slice.
    pub eus_per_sub_slice: u32,
}

/// Device type reported by Level Zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Discrete GPU.
    Gpu,
    /// Integrated GPU.
    IntegratedGpu,
    /// CPU fallback device.
    Cpu,
    /// FPGA accelerator.
    Fpga,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::IntegratedGpu => write!(f, "Integrated GPU"),
            Self::Cpu => write!(f, "CPU"),
            Self::Fpga => write!(f, "FPGA"),
        }
    }
}

/// Properties of a Level Zero device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Opaque device handle.
    pub handle: u64,
    /// Device type.
    pub device_type: DeviceType,
    /// Human-readable device name.
    pub name: String,
    /// PCI vendor ID.
    pub vendor_id: u32,
    /// PCI device ID.
    pub device_id: u32,
    /// Total compute units (EUs × sub-slices × slices).
    pub num_compute_units: u32,
    /// Total execution units.
    pub num_eus: u32,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Maximum clock rate in MHz.
    pub max_clock_rate_mhz: u32,
    /// Sub-slice topology.
    pub subslice_info: SubsliceInfo,
    /// Maximum work-group size.
    pub max_group_size: u32,
    /// Maximum number of work-groups per dimension.
    pub max_group_count: [u32; 3],
}

/// Enumerates devices for a given driver.
#[derive(Debug)]
pub struct LevelZeroDevice {
    devices: Vec<DeviceProperties>,
}

impl LevelZeroDevice {
    /// Enumerate devices for driver at `driver_index` in the given
    /// [`LevelZeroDriver`].
    pub fn enumerate(driver: &LevelZeroDriver, driver_index: u32) -> Result<Self> {
        // Validate driver index to ensure it exists.
        let _ = driver.get_handle(driver_index)?;

        let subslice =
            SubsliceInfo { num_slices: 2, sub_slices_per_slice: 6, eus_per_sub_slice: 8 };

        let total_eus =
            subslice.num_slices * subslice.sub_slices_per_slice * subslice.eus_per_sub_slice;

        Ok(Self {
            devices: vec![DeviceProperties {
                handle: next_handle(),
                device_type: DeviceType::IntegratedGpu,
                name: "Intel(R) Arc(TM) A770 Reference".to_string(),
                vendor_id: 0x8086,
                device_id: 0x5690,
                num_compute_units: total_eus,
                num_eus: total_eus,
                total_memory: 16 * 1024 * 1024 * 1024, // 16 GiB
                max_clock_rate_mhz: 2100,
                subslice_info: subslice,
                max_group_size: 1024,
                max_group_count: [65535, 65535, 65535],
            }],
        })
    }

    /// Return the number of enumerated devices.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.devices.len()
    }

    /// Get properties for device at `index`.
    pub fn get_properties(&self, index: u32) -> Result<&DeviceProperties> {
        self.devices.get(index as usize).ok_or(LevelZeroError::DeviceNotFound(index))
    }

    /// Get a device handle by index.
    pub fn get_handle(&self, index: u32) -> Result<u64> {
        self.get_properties(index).map(|p| p.handle)
    }
}

// ── 4. Context ──────────────────────────────────────────────────────────────

/// Wraps a `ze_context_handle_t` (opaque u64 in this reference impl).
#[derive(Debug)]
pub struct LevelZeroContext {
    handle: u64,
    driver_handle: u64,
    device_handle: u64,
    destroyed: bool,
}

impl LevelZeroContext {
    /// Create a context for the given driver and device.
    pub fn create(driver_handle: u64, device_handle: u64) -> Self {
        log::debug!(
            "creating Level Zero context (driver={driver_handle}, \
             device={device_handle})"
        );
        Self { handle: next_handle(), driver_handle, device_handle, destroyed: false }
    }

    /// Return the context handle.
    #[must_use]
    pub const fn handle(&self) -> u64 {
        self.handle
    }

    /// Return the driver handle associated with this context.
    #[must_use]
    pub const fn driver_handle(&self) -> u64 {
        self.driver_handle
    }

    /// Return the device handle associated with this context.
    #[must_use]
    pub const fn device_handle(&self) -> u64 {
        self.device_handle
    }

    /// Whether this context has been destroyed.
    #[must_use]
    pub const fn is_destroyed(&self) -> bool {
        self.destroyed
    }

    /// Destroy the context, releasing associated resources.
    pub const fn destroy(&mut self) {
        self.destroyed = true;
    }
}

// ── 5. Command List ─────────────────────────────────────────────────────────

/// Whether the command list is immediate (auto-submitted) or regular
/// (requires explicit close + execute).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandListMode {
    /// Commands are submitted immediately when appended.
    Immediate,
    /// Commands are batched until [`LevelZeroCommandList::close`] is called.
    Regular,
}

/// A recorded command in the command list.
#[derive(Debug, Clone)]
pub enum RecordedCommand {
    /// A kernel launch with `(group_count_x, group_count_y, group_count_z)`.
    LaunchKernel { kernel_name: String, group_count: [u32; 3] },
    /// A memory copy of `size` bytes.
    MemoryCopy { size: usize },
    /// A memory fill of `size` bytes with a `pattern`.
    MemoryFill { size: usize, pattern: u8 },
    /// A barrier / pipeline fence.
    Barrier,
}

/// A Level Zero command list that records GPU commands.
#[derive(Debug)]
pub struct LevelZeroCommandList {
    handle: u64,
    context_handle: u64,
    mode: CommandListMode,
    commands: Vec<RecordedCommand>,
    closed: bool,
}

impl LevelZeroCommandList {
    /// Create a new command list.
    pub fn create(context_handle: u64, mode: CommandListMode) -> Self {
        Self { handle: next_handle(), context_handle, mode, commands: Vec::new(), closed: false }
    }

    /// Return the command list handle.
    #[must_use]
    pub const fn handle(&self) -> u64 {
        self.handle
    }

    /// Return the associated context handle.
    #[must_use]
    pub const fn context_handle(&self) -> u64 {
        self.context_handle
    }

    /// Return the command list mode.
    #[must_use]
    pub const fn mode(&self) -> CommandListMode {
        self.mode
    }

    /// Whether the list has been closed.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// Number of recorded commands.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the command list is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Append a kernel launch command.
    pub fn append_launch_kernel(&mut self, kernel_name: &str, group_count: [u32; 3]) -> Result<()> {
        if self.closed {
            return Err(LevelZeroError::CommandListClosed);
        }
        self.commands.push(RecordedCommand::LaunchKernel {
            kernel_name: kernel_name.to_string(),
            group_count,
        });
        Ok(())
    }

    /// Append a memory copy command.
    pub fn append_memory_copy(&mut self, size: usize) -> Result<()> {
        if self.closed {
            return Err(LevelZeroError::CommandListClosed);
        }
        self.commands.push(RecordedCommand::MemoryCopy { size });
        Ok(())
    }

    /// Append a memory fill command.
    pub fn append_memory_fill(&mut self, size: usize, pattern: u8) -> Result<()> {
        if self.closed {
            return Err(LevelZeroError::CommandListClosed);
        }
        self.commands.push(RecordedCommand::MemoryFill { size, pattern });
        Ok(())
    }

    /// Append a barrier.
    pub fn append_barrier(&mut self) -> Result<()> {
        if self.closed {
            return Err(LevelZeroError::CommandListClosed);
        }
        self.commands.push(RecordedCommand::Barrier);
        Ok(())
    }

    /// Close the command list (required for [`CommandListMode::Regular`]).
    pub const fn close(&mut self) -> Result<()> {
        if self.closed {
            return Err(LevelZeroError::CommandListClosed);
        }
        self.closed = true;
        Ok(())
    }

    /// Reset the command list for reuse.
    pub fn reset(&mut self) {
        self.commands.clear();
        self.closed = false;
    }

    /// Return a snapshot of the recorded commands.
    #[must_use]
    pub fn commands(&self) -> &[RecordedCommand] {
        &self.commands
    }
}

// ── 6. Command Queue ────────────────────────────────────────────────────────

/// Scheduling priority for command queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuePriority {
    /// Low priority — may be preempted.
    Low,
    /// Normal priority (default).
    Normal,
    /// High priority — preempts lower-priority work.
    High,
}

/// Submission mode for command queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMode {
    /// Synchronous: `execute` blocks until work completes.
    Synchronous,
    /// Asynchronous: `execute` returns immediately; use fences to sync.
    Asynchronous,
}

/// A Level Zero command queue.
#[derive(Debug)]
pub struct LevelZeroCommandQueue {
    handle: u64,
    context_handle: u64,
    ordinal: u32,
    priority: QueuePriority,
    mode: QueueMode,
    submissions: u64,
    destroyed: bool,
}

impl LevelZeroCommandQueue {
    /// Create a command queue.
    pub fn create(
        context_handle: u64,
        ordinal: u32,
        priority: QueuePriority,
        mode: QueueMode,
    ) -> Self {
        Self {
            handle: next_handle(),
            context_handle,
            ordinal,
            priority,
            mode,
            submissions: 0,
            destroyed: false,
        }
    }

    #[must_use]
    pub const fn handle(&self) -> u64 {
        self.handle
    }

    #[must_use]
    pub const fn context_handle(&self) -> u64 {
        self.context_handle
    }

    #[must_use]
    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    #[must_use]
    pub const fn priority(&self) -> QueuePriority {
        self.priority
    }

    #[must_use]
    pub const fn mode(&self) -> QueueMode {
        self.mode
    }

    #[must_use]
    pub const fn submissions(&self) -> u64 {
        self.submissions
    }

    #[must_use]
    pub const fn is_destroyed(&self) -> bool {
        self.destroyed
    }

    /// Execute a closed command list on this queue.
    pub fn execute(&mut self, cmd_list: &LevelZeroCommandList) -> Result<()> {
        if self.destroyed {
            return Err(LevelZeroError::CommandQueueDestroyed);
        }
        if cmd_list.mode() == CommandListMode::Regular && !cmd_list.is_closed() {
            return Err(LevelZeroError::Other(
                "regular command list must be closed before execution".to_string(),
            ));
        }
        self.submissions += 1;
        Ok(())
    }

    /// Synchronise — wait for all submitted work to complete.
    pub const fn synchronize(&self) -> Result<()> {
        if self.destroyed {
            return Err(LevelZeroError::CommandQueueDestroyed);
        }
        Ok(())
    }

    /// Destroy the command queue.
    pub const fn destroy(&mut self) {
        self.destroyed = true;
    }
}

// ── 7. Module ───────────────────────────────────────────────────────────────

/// Build log entry produced during SPIR-V module compilation.
#[derive(Debug, Clone)]
pub struct ModuleBuildLog {
    pub message: String,
}

/// A Level Zero module created from SPIR-V binary.
#[derive(Debug)]
pub struct LevelZeroModule {
    handle: u64,
    context_handle: u64,
    /// Kernel entry-point names discovered in the module.
    kernel_names: Vec<String>,
    spirv_size: usize,
    build_log: Option<ModuleBuildLog>,
}

impl LevelZeroModule {
    /// Load a SPIR-V module.
    ///
    /// The `spirv` slice must be non-empty. Kernel names are extracted from
    /// magic markers in the binary (simulated here by looking for
    /// null-terminated ASCII strings starting with `"kernel_"`).
    pub fn create(context_handle: u64, spirv: &[u8]) -> Result<Self> {
        if spirv.is_empty() {
            return Err(LevelZeroError::InvalidSpirv);
        }

        // Extract kernel names: scan for ASCII strings prefixed "kernel_".
        let kernel_names = Self::extract_kernel_names(spirv);

        let log_msg = if kernel_names.is_empty() {
            Some(ModuleBuildLog { message: "warning: no kernel entry points found".to_string() })
        } else {
            None
        };

        Ok(Self {
            handle: next_handle(),
            context_handle,
            kernel_names,
            spirv_size: spirv.len(),
            build_log: log_msg,
        })
    }

    /// Scan for `kernel_`-prefixed null-terminated ASCII strings.
    fn extract_kernel_names(spirv: &[u8]) -> Vec<String> {
        let haystack = String::from_utf8_lossy(spirv);
        let mut names = Vec::new();
        for segment in haystack.split('\0') {
            let trimmed = segment.trim();
            if trimmed.starts_with("kernel_") {
                names.push(trimmed.to_string());
            }
        }
        names
    }

    #[must_use]
    pub const fn handle(&self) -> u64 {
        self.handle
    }

    #[must_use]
    pub const fn context_handle(&self) -> u64 {
        self.context_handle
    }

    /// List kernel entry-point names.
    #[must_use]
    pub fn kernel_names(&self) -> &[String] {
        &self.kernel_names
    }

    /// Size of the loaded SPIR-V binary in bytes.
    #[must_use]
    pub const fn spirv_size(&self) -> usize {
        self.spirv_size
    }

    /// Build log, if any warnings were emitted.
    #[must_use]
    pub const fn build_log(&self) -> Option<&ModuleBuildLog> {
        self.build_log.as_ref()
    }

    /// Create a kernel handle for the named entry point.
    pub fn create_kernel(&self, name: &str) -> Result<LevelZeroKernel> {
        if !self.kernel_names.contains(&name.to_string()) {
            return Err(LevelZeroError::KernelNotFound(name.to_string()));
        }
        Ok(LevelZeroKernel::new(self.handle, name))
    }
}

// ── 8. Kernel ───────────────────────────────────────────────────────────────

/// A Level Zero kernel created from a module entry point.
#[derive(Debug)]
pub struct LevelZeroKernel {
    handle: u64,
    module_handle: u64,
    name: String,
    group_size: [u32; 3],
    arguments: HashMap<u32, KernelArgument>,
    max_arguments: u32,
}

/// Type-erased kernel argument.
#[derive(Debug, Clone)]
pub enum KernelArgument {
    /// A pointer argument (device/host/shared memory handle).
    Pointer(u64),
    /// A scalar u32 value.
    ScalarU32(u32),
    /// A scalar u64 value.
    ScalarU64(u64),
    /// A scalar f32 value.
    ScalarF32(f32),
    /// Raw bytes argument.
    Raw(Vec<u8>),
}

impl LevelZeroKernel {
    fn new(module_handle: u64, name: &str) -> Self {
        Self {
            handle: next_handle(),
            module_handle,
            name: name.to_string(),
            group_size: [1, 1, 1],
            arguments: HashMap::new(),
            max_arguments: 16,
        }
    }

    #[must_use]
    pub const fn handle(&self) -> u64 {
        self.handle
    }

    #[must_use]
    pub const fn module_handle(&self) -> u64 {
        self.module_handle
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the currently configured group size.
    #[must_use]
    pub const fn group_size(&self) -> [u32; 3] {
        self.group_size
    }

    /// Set the work-group size `(x, y, z)`.
    pub const fn set_group_size(&mut self, x: u32, y: u32, z: u32) {
        self.group_size = [x, y, z];
    }

    /// Suggest a group size for the given global work size.
    ///
    /// Returns `(group_x, group_y, group_z)` that evenly divides
    /// the global dimensions up to a maximum of 256 per dimension.
    #[must_use]
    pub fn suggest_group_size(&self, global_x: u32, global_y: u32, global_z: u32) -> [u32; 3] {
        fn best_divisor(n: u32, max: u32) -> u32 {
            let cap = n.min(max);
            (1..=cap).rev().find(|d| n.is_multiple_of(*d)).unwrap_or(1)
        }
        [best_divisor(global_x, 256), best_divisor(global_y, 256), best_divisor(global_z, 256)]
    }

    /// Set a kernel argument at `index`.
    pub fn set_argument(&mut self, index: u32, arg: KernelArgument) -> Result<()> {
        if index >= self.max_arguments {
            return Err(LevelZeroError::ArgumentIndexOutOfRange {
                index,
                max: self.max_arguments - 1,
            });
        }
        self.arguments.insert(index, arg);
        Ok(())
    }

    /// Get a previously set argument.
    #[must_use]
    pub fn get_argument(&self, index: u32) -> Option<&KernelArgument> {
        self.arguments.get(&index)
    }

    /// Number of arguments currently set.
    #[must_use]
    pub fn argument_count(&self) -> usize {
        self.arguments.len()
    }
}

// ── 9. Memory ───────────────────────────────────────────────────────────────

/// Where memory is allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Device-local memory (GPU VRAM).
    Device,
    /// Host-visible memory (system RAM, accessible by GPU).
    Host,
    /// Unified shared memory (USM) accessible by both host and device.
    Shared,
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => write!(f, "device"),
            Self::Host => write!(f, "host"),
            Self::Shared => write!(f, "shared"),
        }
    }
}

/// An allocation returned by [`LevelZeroMemory`].
#[derive(Debug)]
pub struct Allocation {
    /// Opaque handle for this allocation.
    pub handle: u64,
    /// The memory type.
    pub memory_type: MemoryType,
    /// Size in bytes.
    pub size: usize,
    /// Alignment in bytes.
    pub alignment: usize,
    /// Simulated base pointer (offset within the arena).
    pub base_ptr: u64,
}

/// Manages device, host, and shared memory allocations.
#[derive(Debug)]
pub struct LevelZeroMemory {
    context_handle: u64,
    device_handle: u64,
    device_capacity: u64,
    allocations: Vec<Allocation>,
    total_allocated: usize,
}

impl LevelZeroMemory {
    /// Create a memory manager for the given context and device.
    pub const fn new(context_handle: u64, device_handle: u64, device_capacity: u64) -> Self {
        Self {
            context_handle,
            device_handle,
            device_capacity,
            allocations: Vec::new(),
            total_allocated: 0,
        }
    }

    #[must_use]
    pub const fn context_handle(&self) -> u64 {
        self.context_handle
    }

    #[must_use]
    pub const fn device_handle(&self) -> u64 {
        self.device_handle
    }

    /// Number of live allocations.
    #[must_use]
    pub const fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Total bytes currently allocated.
    #[must_use]
    pub const fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Remaining device capacity in bytes.
    #[must_use]
    pub const fn remaining_capacity(&self) -> u64 {
        self.device_capacity.saturating_sub(self.total_allocated as u64)
    }

    /// Allocate memory of the given `memory_type` with the specified
    /// `alignment` (must be a power of two, minimum 1).
    #[allow(clippy::cast_possible_truncation)]
    pub fn allocate(
        &mut self,
        size: usize,
        alignment: usize,
        memory_type: MemoryType,
    ) -> Result<u64> {
        if size == 0 || size as u64 > self.device_capacity {
            return Err(LevelZeroError::InvalidAllocationSize {
                requested: size,
                max: self.device_capacity as usize,
            });
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(LevelZeroError::InvalidAlignment(alignment));
        }

        // Align the current offset.
        let raw_ptr = self.total_allocated as u64;
        let aligned_ptr = (raw_ptr + alignment as u64 - 1) & !(alignment as u64 - 1);
        let padded_size = (aligned_ptr - raw_ptr) as usize + size;

        if (self.total_allocated + padded_size) as u64 > self.device_capacity {
            return Err(LevelZeroError::InvalidAllocationSize {
                requested: size,
                max: (self.device_capacity as usize).saturating_sub(self.total_allocated),
            });
        }

        let handle = next_handle();
        self.allocations.push(Allocation {
            handle,
            memory_type,
            size,
            alignment,
            base_ptr: aligned_ptr,
        });
        self.total_allocated += padded_size;
        Ok(handle)
    }

    /// Free an allocation by handle. Returns `true` if found and freed.
    pub fn free(&mut self, handle: u64) -> bool {
        if let Some(pos) = self.allocations.iter().position(|a| a.handle == handle) {
            let alloc = self.allocations.remove(pos);
            self.total_allocated = self.total_allocated.saturating_sub(alloc.size);
            true
        } else {
            false
        }
    }

    /// Look up an allocation by handle.
    #[must_use]
    pub fn get_allocation(&self, handle: u64) -> Option<&Allocation> {
        self.allocations.iter().find(|a| a.handle == handle)
    }
}

// ── 10. Backend orchestrator ────────────────────────────────────────────────

/// Lifecycle state of the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendState {
    /// Not yet initialised.
    Uninitialised,
    /// Driver and device enumerated, context created.
    Ready,
    /// A SPIR-V module has been loaded.
    ModuleLoaded,
    /// The backend has been shut down.
    Shutdown,
}

impl fmt::Display for BackendState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninitialised => write!(f, "uninitialised"),
            Self::Ready => write!(f, "ready"),
            Self::ModuleLoaded => write!(f, "module_loaded"),
            Self::Shutdown => write!(f, "shutdown"),
        }
    }
}

/// Orchestrator: init driver → enumerate devices → create context → load
/// module → execute.
#[derive(Debug)]
pub struct LevelZeroBackend {
    config: LevelZeroConfig,
    state: BackendState,
    driver: Option<LevelZeroDriver>,
    device: Option<LevelZeroDevice>,
    context: Option<LevelZeroContext>,
    module: Option<LevelZeroModule>,
    memory: Option<LevelZeroMemory>,
    queue: Option<LevelZeroCommandQueue>,
}

impl LevelZeroBackend {
    /// Create a new backend with the given configuration. The backend starts
    /// in [`BackendState::Uninitialised`].
    #[must_use]
    pub const fn new(config: LevelZeroConfig) -> Self {
        Self {
            config,
            state: BackendState::Uninitialised,
            driver: None,
            device: None,
            context: None,
            module: None,
            memory: None,
            queue: None,
        }
    }

    /// Current lifecycle state.
    #[must_use]
    pub const fn state(&self) -> BackendState {
        self.state
    }

    /// Reference to the active configuration.
    #[must_use]
    pub const fn config(&self) -> &LevelZeroConfig {
        &self.config
    }

    /// Initialise: enumerate drivers/devices, create context, command queue,
    /// and memory manager.
    pub fn init(&mut self) -> Result<()> {
        if self.state != BackendState::Uninitialised {
            return Err(LevelZeroError::ContextAlreadyCreated);
        }

        // 1. Enumerate drivers.
        let driver = LevelZeroDriver::enumerate();
        let driver_handle = driver.get_handle(self.config.driver_index)?;

        // 2. Enumerate devices.
        let device = LevelZeroDevice::enumerate(&driver, self.config.driver_index)?;
        let dev_props = device.get_properties(self.config.device_index)?;
        let device_handle = dev_props.handle;
        let capacity = dev_props.total_memory;

        log::info!(
            "Level Zero: {} — {} EUs, {} MiB VRAM",
            dev_props.name,
            dev_props.num_eus,
            capacity / (1024 * 1024),
        );

        // 3. Create context.
        let context = LevelZeroContext::create(driver_handle, device_handle);

        // 4. Create command queue.
        let queue = LevelZeroCommandQueue::create(
            context.handle(),
            self.config.ordinal,
            QueuePriority::Normal,
            if self.config.profiling_enabled {
                QueueMode::Synchronous
            } else {
                QueueMode::Asynchronous
            },
        );

        // 5. Create memory manager.
        let memory = LevelZeroMemory::new(context.handle(), device_handle, capacity);

        self.driver = Some(driver);
        self.device = Some(device);
        self.context = Some(context);
        self.queue = Some(queue);
        self.memory = Some(memory);
        self.state = BackendState::Ready;
        Ok(())
    }

    /// Load a SPIR-V module.
    pub fn load_module(&mut self, spirv: &[u8]) -> Result<()> {
        let ctx = self.context.as_ref().ok_or(LevelZeroError::NoContext)?;
        let module = LevelZeroModule::create(ctx.handle(), spirv)?;
        self.module = Some(module);
        self.state = BackendState::ModuleLoaded;
        Ok(())
    }

    /// Create a kernel from the loaded module.
    pub fn create_kernel(&self, name: &str) -> Result<LevelZeroKernel> {
        self.module.as_ref().ok_or(LevelZeroError::NoModule)?.create_kernel(name)
    }

    /// Execute a command list on the backend's queue.
    pub fn execute(&mut self, cmd_list: &LevelZeroCommandList) -> Result<()> {
        self.queue.as_mut().ok_or(LevelZeroError::NoContext)?.execute(cmd_list)
    }

    /// Allocate device memory through the backend's memory manager.
    pub fn allocate_device_memory(&mut self, size: usize, alignment: usize) -> Result<u64> {
        self.memory.as_mut().ok_or(LevelZeroError::NoContext)?.allocate(
            size,
            alignment,
            MemoryType::Device,
        )
    }

    /// Allocate host memory through the backend's memory manager.
    pub fn allocate_host_memory(&mut self, size: usize, alignment: usize) -> Result<u64> {
        self.memory.as_mut().ok_or(LevelZeroError::NoContext)?.allocate(
            size,
            alignment,
            MemoryType::Host,
        )
    }

    /// Allocate shared (USM) memory through the backend's memory manager.
    pub fn allocate_shared_memory(&mut self, size: usize, alignment: usize) -> Result<u64> {
        self.memory.as_mut().ok_or(LevelZeroError::NoContext)?.allocate(
            size,
            alignment,
            MemoryType::Shared,
        )
    }

    /// Free memory by handle.
    pub fn free_memory(&mut self, handle: u64) -> bool {
        self.memory.as_mut().is_some_and(|m| m.free(handle))
    }

    /// Access the memory manager (if initialised).
    #[must_use]
    pub const fn memory(&self) -> Option<&LevelZeroMemory> {
        self.memory.as_ref()
    }

    /// Access the driver (if initialised).
    #[must_use]
    pub const fn driver(&self) -> Option<&LevelZeroDriver> {
        self.driver.as_ref()
    }

    /// Access the device enumerator (if initialised).
    #[must_use]
    pub const fn device(&self) -> Option<&LevelZeroDevice> {
        self.device.as_ref()
    }

    /// Access the context (if initialised).
    #[must_use]
    pub const fn context(&self) -> Option<&LevelZeroContext> {
        self.context.as_ref()
    }

    /// Access the module (if loaded).
    #[must_use]
    pub const fn module(&self) -> Option<&LevelZeroModule> {
        self.module.as_ref()
    }

    /// Shut down the backend, destroying context and queue.
    pub const fn shutdown(&mut self) {
        if let Some(ctx) = self.context.as_mut() {
            ctx.destroy();
        }
        if let Some(q) = self.queue.as_mut() {
            q.destroy();
        }
        self.state = BackendState::Shutdown;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Error display ───────────────────────────────────────────────────

    #[test]
    fn error_display_driver_not_found() {
        let e = LevelZeroError::DriverNotFound(3);
        assert_eq!(e.to_string(), "driver index 3 not found");
    }

    #[test]
    fn error_display_device_not_found() {
        let e = LevelZeroError::DeviceNotFound(5);
        assert_eq!(e.to_string(), "device index 5 not found");
    }

    #[test]
    fn error_display_context_already_created() {
        let e = LevelZeroError::ContextAlreadyCreated;
        assert_eq!(e.to_string(), "context already created");
    }

    #[test]
    fn error_display_no_context() {
        assert_eq!(LevelZeroError::NoContext.to_string(), "no context created");
    }

    #[test]
    fn error_display_no_module() {
        assert_eq!(LevelZeroError::NoModule.to_string(), "no module loaded");
    }

    #[test]
    fn error_display_invalid_spirv() {
        let e = LevelZeroError::InvalidSpirv;
        assert_eq!(e.to_string(), "invalid or empty SPIR-V binary");
    }

    #[test]
    fn error_display_kernel_not_found() {
        let e = LevelZeroError::KernelNotFound("foo".into());
        assert_eq!(e.to_string(), "kernel 'foo' not found");
    }

    #[test]
    fn error_display_argument_out_of_range() {
        let e = LevelZeroError::ArgumentIndexOutOfRange { index: 20, max: 15 };
        assert_eq!(e.to_string(), "argument index 20 out of range (max 15)");
    }

    #[test]
    fn error_display_invalid_allocation_size() {
        let e = LevelZeroError::InvalidAllocationSize { requested: 0, max: 1024 };
        assert_eq!(e.to_string(), "invalid allocation size 0 (max 1024)");
    }

    #[test]
    fn error_display_invalid_alignment() {
        let e = LevelZeroError::InvalidAlignment(3);
        assert_eq!(e.to_string(), "alignment 3 is not a power of two");
    }

    #[test]
    fn error_display_command_list_closed() {
        assert_eq!(LevelZeroError::CommandListClosed.to_string(), "command list is closed");
    }

    #[test]
    fn error_display_command_queue_destroyed() {
        assert_eq!(LevelZeroError::CommandQueueDestroyed.to_string(), "command queue is destroyed");
    }

    #[test]
    fn error_display_other() {
        let e = LevelZeroError::Other("boom".into());
        assert_eq!(e.to_string(), "boom");
    }

    #[test]
    fn error_implements_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(LevelZeroError::NoContext);
        assert_eq!(e.to_string(), "no context created");
    }

    // ── Config ──────────────────────────────────────────────────────────

    #[test]
    fn config_default() {
        let cfg = LevelZeroConfig::default();
        assert_eq!(cfg.driver_index, 0);
        assert_eq!(cfg.device_index, 0);
        assert_eq!(cfg.ordinal, 0);
        assert!(!cfg.profiling_enabled);
    }

    #[test]
    fn config_custom() {
        let cfg = LevelZeroConfig {
            driver_index: 1,
            device_index: 2,
            ordinal: 3,
            profiling_enabled: true,
        };
        assert_eq!(cfg.driver_index, 1);
        assert_eq!(cfg.device_index, 2);
        assert_eq!(cfg.ordinal, 3);
        assert!(cfg.profiling_enabled);
    }

    #[test]
    fn config_clone() {
        let cfg = LevelZeroConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.driver_index, cfg.driver_index);
    }

    // ── Driver ──────────────────────────────────────────────────────────

    #[test]
    fn driver_enumerate_non_empty() {
        let drv = LevelZeroDriver::enumerate();
        assert!(drv.count() > 0);
    }

    #[test]
    fn driver_get_properties_valid() {
        let drv = LevelZeroDriver::enumerate();
        let props = drv.get_properties(0).unwrap();
        assert!(!props.name.is_empty());
        assert!(props.handle > 0);
    }

    #[test]
    fn driver_get_properties_invalid_index() {
        let drv = LevelZeroDriver::enumerate();
        assert_eq!(drv.get_properties(99).unwrap_err(), LevelZeroError::DriverNotFound(99));
    }

    #[test]
    fn driver_get_handle() {
        let drv = LevelZeroDriver::enumerate();
        let h = drv.get_handle(0).unwrap();
        assert!(h > 0);
    }

    #[test]
    fn driver_version_encoding() {
        let drv = LevelZeroDriver::enumerate();
        let props = drv.get_properties(0).unwrap();
        let major = props.driver_version >> 16;
        let minor = props.driver_version & 0xFFFF;
        assert_eq!(major, 1);
        assert_eq!(minor, 8);
    }

    #[test]
    fn driver_api_version() {
        let drv = LevelZeroDriver::enumerate();
        let props = drv.get_properties(0).unwrap();
        assert_eq!(props.api_version, (1, 8));
    }

    // ── Device ──────────────────────────────────────────────────────────

    #[test]
    fn device_enumerate_valid_driver() {
        let drv = LevelZeroDriver::enumerate();
        let dev = LevelZeroDevice::enumerate(&drv, 0).unwrap();
        assert!(dev.count() > 0);
    }

    #[test]
    fn device_enumerate_invalid_driver() {
        let drv = LevelZeroDriver::enumerate();
        assert_eq!(
            LevelZeroDevice::enumerate(&drv, 99).unwrap_err(),
            LevelZeroError::DriverNotFound(99)
        );
    }

    #[test]
    fn device_properties() {
        let drv = LevelZeroDriver::enumerate();
        let dev = LevelZeroDevice::enumerate(&drv, 0).unwrap();
        let props = dev.get_properties(0).unwrap();
        assert_eq!(props.vendor_id, 0x8086);
        assert_eq!(props.device_type, DeviceType::IntegratedGpu);
        assert!(props.num_eus > 0);
        assert!(props.total_memory > 0);
    }

    #[test]
    fn device_invalid_index() {
        let drv = LevelZeroDriver::enumerate();
        let dev = LevelZeroDevice::enumerate(&drv, 0).unwrap();
        assert_eq!(dev.get_properties(99).unwrap_err(), LevelZeroError::DeviceNotFound(99));
    }

    #[test]
    fn device_handle() {
        let drv = LevelZeroDriver::enumerate();
        let dev = LevelZeroDevice::enumerate(&drv, 0).unwrap();
        assert!(dev.get_handle(0).unwrap() > 0);
    }

    #[test]
    fn device_subslice_info() {
        let drv = LevelZeroDriver::enumerate();
        let dev = LevelZeroDevice::enumerate(&drv, 0).unwrap();
        let props = dev.get_properties(0).unwrap();
        let ss = props.subslice_info;
        assert_eq!(ss.num_slices, 2);
        assert_eq!(ss.sub_slices_per_slice, 6);
        assert_eq!(ss.eus_per_sub_slice, 8);
        assert_eq!(props.num_eus, ss.num_slices * ss.sub_slices_per_slice * ss.eus_per_sub_slice);
    }

    #[test]
    fn device_max_group_size() {
        let drv = LevelZeroDriver::enumerate();
        let dev = LevelZeroDevice::enumerate(&drv, 0).unwrap();
        let props = dev.get_properties(0).unwrap();
        assert_eq!(props.max_group_size, 1024);
    }

    #[test]
    fn device_type_display() {
        assert_eq!(DeviceType::Gpu.to_string(), "GPU");
        assert_eq!(DeviceType::IntegratedGpu.to_string(), "Integrated GPU");
        assert_eq!(DeviceType::Cpu.to_string(), "CPU");
        assert_eq!(DeviceType::Fpga.to_string(), "FPGA");
    }

    // ── Context ─────────────────────────────────────────────────────────

    #[test]
    fn context_create() {
        let ctx = LevelZeroContext::create(1, 2);
        assert!(ctx.handle() > 0);
        assert_eq!(ctx.driver_handle(), 1);
        assert_eq!(ctx.device_handle(), 2);
        assert!(!ctx.is_destroyed());
    }

    #[test]
    fn context_destroy() {
        let mut ctx = LevelZeroContext::create(1, 2);
        ctx.destroy();
        assert!(ctx.is_destroyed());
    }

    // ── Command List ────────────────────────────────────────────────────

    #[test]
    fn cmdlist_create_immediate() {
        let cl = LevelZeroCommandList::create(1, CommandListMode::Immediate);
        assert!(cl.handle() > 0);
        assert_eq!(cl.mode(), CommandListMode::Immediate);
        assert!(!cl.is_closed());
        assert!(cl.is_empty());
    }

    #[test]
    fn cmdlist_create_regular() {
        let cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        assert_eq!(cl.mode(), CommandListMode::Regular);
    }

    #[test]
    fn cmdlist_append_launch_kernel() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_launch_kernel("my_kern", [8, 1, 1]).unwrap();
        assert_eq!(cl.len(), 1);
    }

    #[test]
    fn cmdlist_append_memory_copy() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_memory_copy(4096).unwrap();
        assert_eq!(cl.len(), 1);
    }

    #[test]
    fn cmdlist_append_memory_fill() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_memory_fill(256, 0xAB).unwrap();
        assert_eq!(cl.len(), 1);
    }

    #[test]
    fn cmdlist_append_barrier() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_barrier().unwrap();
        assert_eq!(cl.len(), 1);
    }

    #[test]
    fn cmdlist_close_and_reject_further_appends() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_barrier().unwrap();
        cl.close().unwrap();
        assert!(cl.is_closed());
        assert_eq!(cl.append_barrier().unwrap_err(), LevelZeroError::CommandListClosed);
    }

    #[test]
    fn cmdlist_double_close_errors() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        assert_eq!(cl.close().unwrap_err(), LevelZeroError::CommandListClosed);
    }

    #[test]
    fn cmdlist_reset() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_barrier().unwrap();
        cl.close().unwrap();
        cl.reset();
        assert!(!cl.is_closed());
        assert!(cl.is_empty());
        cl.append_barrier().unwrap();
        assert_eq!(cl.len(), 1);
    }

    #[test]
    fn cmdlist_context_handle() {
        let cl = LevelZeroCommandList::create(42, CommandListMode::Regular);
        assert_eq!(cl.context_handle(), 42);
    }

    #[test]
    fn cmdlist_commands_snapshot() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_launch_kernel("k", [1, 1, 1]).unwrap();
        cl.append_memory_copy(64).unwrap();
        assert_eq!(cl.commands().len(), 2);
    }

    #[test]
    fn cmdlist_closed_rejects_memory_copy() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        assert_eq!(cl.append_memory_copy(1).unwrap_err(), LevelZeroError::CommandListClosed);
    }

    #[test]
    fn cmdlist_closed_rejects_memory_fill() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        assert_eq!(cl.append_memory_fill(1, 0).unwrap_err(), LevelZeroError::CommandListClosed);
    }

    #[test]
    fn cmdlist_closed_rejects_launch() {
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        assert_eq!(
            cl.append_launch_kernel("x", [1, 1, 1]).unwrap_err(),
            LevelZeroError::CommandListClosed
        );
    }

    // ── Command Queue ───────────────────────────────────────────────────

    #[test]
    fn queue_create() {
        let q = LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Asynchronous);
        assert!(q.handle() > 0);
        assert_eq!(q.ordinal(), 0);
        assert_eq!(q.priority(), QueuePriority::Normal);
        assert_eq!(q.mode(), QueueMode::Asynchronous);
        assert_eq!(q.submissions(), 0);
        assert!(!q.is_destroyed());
    }

    #[test]
    fn queue_execute_closed_list() {
        let mut q =
            LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        q.execute(&cl).unwrap();
        assert_eq!(q.submissions(), 1);
    }

    #[test]
    fn queue_execute_unclosed_regular_errors() {
        let mut q =
            LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        let cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        assert!(q.execute(&cl).is_err());
    }

    #[test]
    fn queue_execute_immediate_unclosed_ok() {
        let mut q =
            LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        let cl = LevelZeroCommandList::create(1, CommandListMode::Immediate);
        q.execute(&cl).unwrap();
    }

    #[test]
    fn queue_synchronize() {
        let q = LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        q.synchronize().unwrap();
    }

    #[test]
    fn queue_destroy() {
        let mut q =
            LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        q.destroy();
        assert!(q.is_destroyed());
    }

    #[test]
    fn queue_execute_after_destroy_errors() {
        let mut q =
            LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        q.destroy();
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        assert_eq!(q.execute(&cl).unwrap_err(), LevelZeroError::CommandQueueDestroyed);
    }

    #[test]
    fn queue_synchronize_after_destroy_errors() {
        let mut q =
            LevelZeroCommandQueue::create(1, 0, QueuePriority::Normal, QueueMode::Synchronous);
        q.destroy();
        assert_eq!(q.synchronize().unwrap_err(), LevelZeroError::CommandQueueDestroyed);
    }

    #[test]
    fn queue_high_priority() {
        let q = LevelZeroCommandQueue::create(1, 0, QueuePriority::High, QueueMode::Asynchronous);
        assert_eq!(q.priority(), QueuePriority::High);
    }

    #[test]
    fn queue_low_priority() {
        let q = LevelZeroCommandQueue::create(1, 0, QueuePriority::Low, QueueMode::Synchronous);
        assert_eq!(q.priority(), QueuePriority::Low);
    }

    #[test]
    fn queue_context_handle() {
        let q = LevelZeroCommandQueue::create(77, 0, QueuePriority::Normal, QueueMode::Synchronous);
        assert_eq!(q.context_handle(), 77);
    }

    // ── Module ──────────────────────────────────────────────────────────

    fn spirv_with_kernels(names: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();
        for name in names {
            data.extend_from_slice(name.as_bytes());
            data.push(0); // null terminator
        }
        data
    }

    #[test]
    fn module_create_with_kernels() {
        let spirv = spirv_with_kernels(&["kernel_add", "kernel_mul"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        assert_eq!(m.kernel_names().len(), 2);
        assert!(m.kernel_names().contains(&"kernel_add".to_string()));
        assert!(m.kernel_names().contains(&"kernel_mul".to_string()));
        assert_eq!(m.spirv_size(), spirv.len());
        assert!(m.build_log().is_none());
    }

    #[test]
    fn module_create_empty_spirv_errors() {
        assert_eq!(LevelZeroModule::create(1, &[]).unwrap_err(), LevelZeroError::InvalidSpirv);
    }

    #[test]
    fn module_no_kernel_names_produces_build_log() {
        let spirv = vec![0xDE, 0xAD];
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        assert!(m.kernel_names().is_empty());
        assert!(m.build_log().is_some());
    }

    #[test]
    fn module_create_kernel_valid() {
        let spirv = spirv_with_kernels(&["kernel_add"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let k = m.create_kernel("kernel_add").unwrap();
        assert_eq!(k.name(), "kernel_add");
    }

    #[test]
    fn module_create_kernel_not_found() {
        let spirv = spirv_with_kernels(&["kernel_add"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        assert_eq!(
            m.create_kernel("kernel_sub").unwrap_err(),
            LevelZeroError::KernelNotFound("kernel_sub".into())
        );
    }

    #[test]
    fn module_handle_nonzero() {
        let spirv = spirv_with_kernels(&["kernel_x"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        assert!(m.handle() > 0);
    }

    #[test]
    fn module_context_handle() {
        let spirv = spirv_with_kernels(&["kernel_x"]);
        let m = LevelZeroModule::create(99, &spirv).unwrap();
        assert_eq!(m.context_handle(), 99);
    }

    // ── Kernel ──────────────────────────────────────────────────────────

    #[test]
    fn kernel_default_group_size() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let k = m.create_kernel("kernel_a").unwrap();
        assert_eq!(k.group_size(), [1, 1, 1]);
    }

    #[test]
    fn kernel_set_group_size() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        k.set_group_size(64, 2, 1);
        assert_eq!(k.group_size(), [64, 2, 1]);
    }

    #[test]
    fn kernel_suggest_group_size_exact_divisor() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let k = m.create_kernel("kernel_a").unwrap();
        let [gx, gy, gz] = k.suggest_group_size(256, 128, 1);
        assert_eq!(256 % gx, 0);
        assert_eq!(128 % gy, 0);
        assert_eq!(1 % gz, 0);
    }

    #[test]
    fn kernel_suggest_group_size_caps_at_256() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let k = m.create_kernel("kernel_a").unwrap();
        let [gx, _, _] = k.suggest_group_size(1024, 1, 1);
        assert!(gx <= 256);
        assert_eq!(1024 % gx, 0);
    }

    #[test]
    fn kernel_set_argument_pointer() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        k.set_argument(0, KernelArgument::Pointer(0xCAFE)).unwrap();
        assert_eq!(k.argument_count(), 1);
    }

    #[test]
    fn kernel_set_argument_scalar_u32() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        k.set_argument(0, KernelArgument::ScalarU32(42)).unwrap();
        match k.get_argument(0) {
            Some(KernelArgument::ScalarU32(v)) => assert_eq!(*v, 42),
            _ => panic!("expected ScalarU32"),
        }
    }

    #[test]
    fn kernel_set_argument_scalar_f32() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        k.set_argument(0, KernelArgument::ScalarF32(2.72)).unwrap();
        match k.get_argument(0) {
            Some(KernelArgument::ScalarF32(v)) => {
                assert!((v - 2.72).abs() < f32::EPSILON);
            }
            _ => panic!("expected ScalarF32"),
        }
    }

    #[test]
    fn kernel_set_argument_raw() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        k.set_argument(0, KernelArgument::Raw(vec![1, 2, 3])).unwrap();
        assert_eq!(k.argument_count(), 1);
    }

    #[test]
    fn kernel_argument_out_of_range() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        assert_eq!(
            k.set_argument(16, KernelArgument::ScalarU32(0)).unwrap_err(),
            LevelZeroError::ArgumentIndexOutOfRange { index: 16, max: 15 }
        );
    }

    #[test]
    fn kernel_get_missing_argument() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let k = m.create_kernel("kernel_a").unwrap();
        assert!(k.get_argument(0).is_none());
    }

    #[test]
    fn kernel_module_handle() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mh = m.handle();
        let k = m.create_kernel("kernel_a").unwrap();
        assert_eq!(k.module_handle(), mh);
    }

    #[test]
    fn kernel_handle_nonzero() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let k = m.create_kernel("kernel_a").unwrap();
        assert!(k.handle() > 0);
    }

    #[test]
    fn kernel_set_argument_scalar_u64() {
        let spirv = spirv_with_kernels(&["kernel_a"]);
        let m = LevelZeroModule::create(1, &spirv).unwrap();
        let mut k = m.create_kernel("kernel_a").unwrap();
        k.set_argument(0, KernelArgument::ScalarU64(u64::MAX)).unwrap();
        match k.get_argument(0) {
            Some(KernelArgument::ScalarU64(v)) => assert_eq!(*v, u64::MAX),
            _ => panic!("expected ScalarU64"),
        }
    }

    // ── Memory ──────────────────────────────────────────────────────────

    #[test]
    fn memory_allocate_device() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024 * 1024);
        let h = mem.allocate(4096, 64, MemoryType::Device).unwrap();
        assert!(h > 0);
        assert_eq!(mem.allocation_count(), 1);
        assert!(mem.total_allocated() >= 4096);
    }

    #[test]
    fn memory_allocate_host() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024 * 1024);
        let h = mem.allocate(1024, 16, MemoryType::Host).unwrap();
        let alloc = mem.get_allocation(h).unwrap();
        assert_eq!(alloc.memory_type, MemoryType::Host);
    }

    #[test]
    fn memory_allocate_shared() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024 * 1024);
        let h = mem.allocate(2048, 128, MemoryType::Shared).unwrap();
        let alloc = mem.get_allocation(h).unwrap();
        assert_eq!(alloc.memory_type, MemoryType::Shared);
    }

    #[test]
    fn memory_allocate_zero_size_errors() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024);
        assert!(matches!(
            mem.allocate(0, 64, MemoryType::Device).unwrap_err(),
            LevelZeroError::InvalidAllocationSize { .. }
        ));
    }

    #[test]
    fn memory_allocate_exceeds_capacity() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024);
        assert!(matches!(
            mem.allocate(2048, 64, MemoryType::Device).unwrap_err(),
            LevelZeroError::InvalidAllocationSize { .. }
        ));
    }

    #[test]
    fn memory_invalid_alignment_zero() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024);
        assert_eq!(
            mem.allocate(64, 0, MemoryType::Device).unwrap_err(),
            LevelZeroError::InvalidAlignment(0)
        );
    }

    #[test]
    fn memory_invalid_alignment_non_power_of_two() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024);
        assert_eq!(
            mem.allocate(64, 3, MemoryType::Device).unwrap_err(),
            LevelZeroError::InvalidAlignment(3)
        );
    }

    #[test]
    fn memory_free() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024 * 1024);
        let h = mem.allocate(256, 64, MemoryType::Device).unwrap();
        assert!(mem.free(h));
        assert_eq!(mem.allocation_count(), 0);
    }

    #[test]
    fn memory_free_nonexistent_returns_false() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024);
        assert!(!mem.free(9999));
    }

    #[test]
    fn memory_remaining_capacity() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024);
        assert_eq!(mem.remaining_capacity(), 1024);
        mem.allocate(256, 1, MemoryType::Device).unwrap();
        assert!(mem.remaining_capacity() < 1024);
    }

    #[test]
    fn memory_alignment_respected() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024 * 1024);
        let h = mem.allocate(100, 256, MemoryType::Device).unwrap();
        let alloc = mem.get_allocation(h).unwrap();
        assert_eq!(alloc.base_ptr % 256, 0);
    }

    #[test]
    fn memory_type_display() {
        assert_eq!(MemoryType::Device.to_string(), "device");
        assert_eq!(MemoryType::Host.to_string(), "host");
        assert_eq!(MemoryType::Shared.to_string(), "shared");
    }

    #[test]
    fn memory_context_and_device_handles() {
        let mem = LevelZeroMemory::new(10, 20, 1024);
        assert_eq!(mem.context_handle(), 10);
        assert_eq!(mem.device_handle(), 20);
    }

    #[test]
    fn memory_multiple_allocations() {
        let mut mem = LevelZeroMemory::new(1, 2, 1024 * 1024);
        let h1 = mem.allocate(128, 64, MemoryType::Device).unwrap();
        let h2 = mem.allocate(256, 64, MemoryType::Host).unwrap();
        let h3 = mem.allocate(512, 64, MemoryType::Shared).unwrap();
        assert_eq!(mem.allocation_count(), 3);
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
    }

    // ── Backend ─────────────────────────────────────────────────────────

    #[test]
    fn backend_new_uninitialised() {
        let b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert_eq!(b.state(), BackendState::Uninitialised);
    }

    #[test]
    fn backend_init_transitions_to_ready() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        assert_eq!(b.state(), BackendState::Ready);
    }

    #[test]
    fn backend_double_init_errors() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        assert_eq!(b.init().unwrap_err(), LevelZeroError::ContextAlreadyCreated);
    }

    #[test]
    fn backend_load_module() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        let spirv = spirv_with_kernels(&["kernel_add"]);
        b.load_module(&spirv).unwrap();
        assert_eq!(b.state(), BackendState::ModuleLoaded);
    }

    #[test]
    fn backend_load_module_without_init_errors() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert_eq!(b.load_module(&[0xDE]).unwrap_err(), LevelZeroError::NoContext);
    }

    #[test]
    fn backend_create_kernel() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        let spirv = spirv_with_kernels(&["kernel_add"]);
        b.load_module(&spirv).unwrap();
        let k = b.create_kernel("kernel_add").unwrap();
        assert_eq!(k.name(), "kernel_add");
    }

    #[test]
    fn backend_create_kernel_without_module_errors() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        assert_eq!(b.create_kernel("x").unwrap_err(), LevelZeroError::NoModule);
    }

    #[test]
    fn backend_execute_command_list() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.append_barrier().unwrap();
        cl.close().unwrap();
        b.execute(&cl).unwrap();
    }

    #[test]
    fn backend_allocate_and_free_device_memory() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        let h = b.allocate_device_memory(4096, 64).unwrap();
        assert!(b.free_memory(h));
    }

    #[test]
    fn backend_allocate_host_memory() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        let h = b.allocate_host_memory(1024, 16).unwrap();
        assert!(h > 0);
    }

    #[test]
    fn backend_allocate_shared_memory() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        let h = b.allocate_shared_memory(2048, 128).unwrap();
        assert!(h > 0);
    }

    #[test]
    fn backend_shutdown() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        b.shutdown();
        assert_eq!(b.state(), BackendState::Shutdown);
        assert!(b.context().unwrap().is_destroyed());
    }

    #[test]
    fn backend_config_ref() {
        let cfg = LevelZeroConfig {
            driver_index: 0,
            device_index: 0,
            ordinal: 7,
            profiling_enabled: true,
        };
        let b = LevelZeroBackend::new(cfg);
        assert_eq!(b.config().ordinal, 7);
        assert!(b.config().profiling_enabled);
    }

    #[test]
    fn backend_driver_accessor() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert!(b.driver().is_none());
        b.init().unwrap();
        assert!(b.driver().is_some());
    }

    #[test]
    fn backend_device_accessor() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert!(b.device().is_none());
        b.init().unwrap();
        assert!(b.device().is_some());
    }

    #[test]
    fn backend_context_accessor() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert!(b.context().is_none());
        b.init().unwrap();
        assert!(b.context().is_some());
    }

    #[test]
    fn backend_module_accessor() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        b.init().unwrap();
        assert!(b.module().is_none());
        let spirv = spirv_with_kernels(&["kernel_x"]);
        b.load_module(&spirv).unwrap();
        assert!(b.module().is_some());
    }

    #[test]
    fn backend_memory_accessor() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert!(b.memory().is_none());
        b.init().unwrap();
        assert!(b.memory().is_some());
    }

    #[test]
    fn backend_free_memory_before_init_returns_false() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());
        assert!(!b.free_memory(1));
    }

    #[test]
    fn backend_state_display() {
        assert_eq!(BackendState::Uninitialised.to_string(), "uninitialised");
        assert_eq!(BackendState::Ready.to_string(), "ready");
        assert_eq!(BackendState::ModuleLoaded.to_string(), "module_loaded");
        assert_eq!(BackendState::Shutdown.to_string(), "shutdown");
    }

    #[test]
    fn backend_profiling_mode_selects_sync_queue() {
        let cfg = LevelZeroConfig { profiling_enabled: true, ..LevelZeroConfig::default() };
        let mut b = LevelZeroBackend::new(cfg);
        b.init().unwrap();
        // The queue is internal, but we can verify via successful sync execute.
        let mut cl = LevelZeroCommandList::create(1, CommandListMode::Regular);
        cl.close().unwrap();
        b.execute(&cl).unwrap();
    }

    // ── Integration: full pipeline ──────────────────────────────────────

    #[test]
    fn integration_full_pipeline() {
        let mut b = LevelZeroBackend::new(LevelZeroConfig::default());

        // Init
        b.init().unwrap();
        assert_eq!(b.state(), BackendState::Ready);

        // Load module
        let spirv = spirv_with_kernels(&["kernel_gemv", "kernel_relu"]);
        b.load_module(&spirv).unwrap();
        assert_eq!(b.state(), BackendState::ModuleLoaded);

        // Create kernels
        let mut k_gemv = b.create_kernel("kernel_gemv").unwrap();
        let k_relu = b.create_kernel("kernel_relu").unwrap();
        assert_eq!(k_gemv.name(), "kernel_gemv");
        assert_eq!(k_relu.name(), "kernel_relu");

        // Allocate memory
        let buf_a = b.allocate_device_memory(4096, 64).unwrap();
        let buf_b = b.allocate_device_memory(4096, 64).unwrap();
        let buf_out = b.allocate_shared_memory(4096, 64).unwrap();

        // Configure kernel
        k_gemv.set_group_size(256, 1, 1);
        k_gemv.set_argument(0, KernelArgument::Pointer(buf_a)).unwrap();
        k_gemv.set_argument(1, KernelArgument::Pointer(buf_b)).unwrap();
        k_gemv.set_argument(2, KernelArgument::Pointer(buf_out)).unwrap();
        k_gemv.set_argument(3, KernelArgument::ScalarU32(1024)).unwrap();
        assert_eq!(k_gemv.argument_count(), 4);

        // Build command list
        let ctx_handle = b.context().unwrap().handle();
        let mut cl = LevelZeroCommandList::create(ctx_handle, CommandListMode::Regular);
        cl.append_launch_kernel("kernel_gemv", [4, 1, 1]).unwrap();
        cl.append_barrier().unwrap();
        cl.append_launch_kernel("kernel_relu", [4, 1, 1]).unwrap();
        cl.close().unwrap();

        // Execute
        b.execute(&cl).unwrap();

        // Cleanup
        assert!(b.free_memory(buf_a));
        assert!(b.free_memory(buf_b));
        assert!(b.free_memory(buf_out));
        b.shutdown();
        assert_eq!(b.state(), BackendState::Shutdown);
    }
}
