//! Module stub - implementation pending merge from feature branch
//! Vulkan compute pipeline for GPU kernel execution.
//!
//! CPU reference implementation simulating the Vulkan API surface.
//! All types have correct API shape but use CPU fallback internally.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can occur during Vulkan compute operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanError {
    /// Vulkan instance creation failed.
    InstanceCreationFailed(String),
    /// No suitable physical device found.
    NoSuitableDevice,
    /// No compute-capable queue family found.
    NoComputeQueueFamily,
    /// Buffer allocation failed.
    BufferAllocationFailed(String),
    /// Shader module creation failed (e.g. invalid SPIR-V).
    InvalidShaderModule(String),
    /// Pipeline creation failed.
    PipelineCreationFailed(String),
    /// Command buffer recording error.
    CommandBufferError(String),
    /// Fence wait timed out.
    FenceTimeout,
    /// Dispatch with zero work groups.
    ZeroWorkGroups,
    /// Push constant size exceeds device limit.
    PushConstantOverflow { actual: usize, limit: usize },
    /// Buffer index out of range for descriptor set.
    DescriptorBindingOutOfRange { binding: u32, max: u32 },
    /// Device lost (simulated).
    DeviceLost,
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InstanceCreationFailed(msg) => {
                write!(f, "Vulkan instance creation failed: {msg}")
            }
            Self::NoSuitableDevice => write!(f, "no suitable physical device"),
            Self::NoComputeQueueFamily => {
                write!(f, "no compute-capable queue family")
            }
            Self::BufferAllocationFailed(msg) => {
                write!(f, "buffer allocation failed: {msg}")
            }
            Self::InvalidShaderModule(msg) => {
                write!(f, "invalid shader module: {msg}")
            }
            Self::PipelineCreationFailed(msg) => {
                write!(f, "pipeline creation failed: {msg}")
            }
            Self::CommandBufferError(msg) => {
                write!(f, "command buffer error: {msg}")
            }
            Self::FenceTimeout => write!(f, "fence wait timed out"),
            Self::ZeroWorkGroups => write!(f, "dispatch with zero work groups"),
            Self::PushConstantOverflow { actual, limit } => {
                write!(f, "push constant size {actual} exceeds limit {limit}")
            }
            Self::DescriptorBindingOutOfRange { binding, max } => {
                write!(f, "descriptor binding {binding} out of range (max {max})")
            }
            Self::DeviceLost => write!(f, "device lost"),
        }
    }
}

impl std::error::Error for VulkanError {}

pub type Result<T> = std::result::Result<T, VulkanError>;

// ── Configuration ───────────────────────────────────────────────────────────

/// Configuration for Vulkan instance and device creation.
#[derive(Debug, Clone)]
pub struct VulkanConfig {
    /// Enable Vulkan validation layers (debug only).
    pub enable_validation_layers: bool,
    /// Preferred physical device index (0 = first).
    pub device_index: u32,
    /// Preferred queue family index for compute.
    /// `None` means auto-select the first compute-capable family.
    pub queue_family_index: Option<u32>,
    /// Maximum push constant size in bytes.
    pub max_push_constant_size: usize,
    /// Application name reported to the Vulkan loader.
    pub application_name: String,
    /// Application version (packed Vulkan-style).
    pub application_version: u32,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        Self {
            enable_validation_layers: false,
            device_index: 0,
            queue_family_index: None,
            max_push_constant_size: 128,
            application_name: "bitnet-gpu-hal".to_string(),
            application_version: 1,
        }
    }
}

// ── Vulkan Instance ─────────────────────────────────────────────────────────

/// Simulated debug callback message.
#[derive(Debug, Clone)]
pub struct DebugMessage {
    pub severity: DebugSeverity,
    pub message: String,
}

/// Severity levels for debug messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugSeverity {
    Info,
    Warning,
    Error,
}

/// Wraps a `VkInstance` with optional validation layers and debug callback.
#[derive(Debug)]
pub struct VulkanInstance {
    config: VulkanConfig,
    validation_enabled: bool,
    debug_messages: Vec<DebugMessage>,
    created: bool,
}

impl VulkanInstance {
    /// Create a new Vulkan instance.
    pub fn new(config: &VulkanConfig) -> Result<Self> {
        if config.application_name.is_empty() {
            return Err(VulkanError::InstanceCreationFailed(
                "application name must not be empty".to_string(),
            ));
        }
        let mut inst = Self {
            config: config.clone(),
            validation_enabled: config.enable_validation_layers,
            debug_messages: Vec::new(),
            created: true,
        };
        if inst.validation_enabled {
            inst.push_debug(DebugSeverity::Info, "validation layers enabled".to_string());
        }
        Ok(inst)
    }

    /// Whether validation layers are active.
    pub const fn validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// Drain accumulated debug messages.
    pub fn drain_debug_messages(&mut self) -> Vec<DebugMessage> {
        std::mem::take(&mut self.debug_messages)
    }

    /// Reference to the config used to create this instance.
    pub const fn config(&self) -> &VulkanConfig {
        &self.config
    }

    /// Whether the instance was successfully created.
    pub const fn is_created(&self) -> bool {
        self.created
    }

    fn push_debug(&mut self, severity: DebugSeverity, message: String) {
        self.debug_messages.push(DebugMessage { severity, message });
    }
}

// ── Physical / Logical Device ───────────────────────────────────────────────

/// Properties of a simulated physical device.
#[derive(Debug, Clone)]
pub struct PhysicalDeviceProperties {
    pub device_name: String,
    pub device_type: PhysicalDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_size: [u32; 3],
    pub max_push_constant_size: usize,
}

/// Vulkan physical device type categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalDeviceType {
    DiscreteGpu,
    IntegratedGpu,
    Cpu,
    Other,
}

impl fmt::Display for PhysicalDeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DiscreteGpu => write!(f, "Discrete GPU"),
            Self::IntegratedGpu => write!(f, "Integrated GPU"),
            Self::Cpu => write!(f, "CPU"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Queue family properties.
#[derive(Debug, Clone)]
pub struct QueueFamilyProperties {
    pub index: u32,
    pub queue_count: u32,
    pub supports_compute: bool,
    pub supports_transfer: bool,
    pub supports_graphics: bool,
}

/// Wraps physical device selection, logical device, and compute queue family.
#[derive(Debug)]
pub struct VulkanDevice {
    physical_properties: PhysicalDeviceProperties,
    queue_families: Vec<QueueFamilyProperties>,
    selected_queue_family: u32,
    max_push_constant_size: usize,
}

impl VulkanDevice {
    /// Enumerate simulated physical devices.
    pub fn enumerate_physical_devices() -> Vec<PhysicalDeviceProperties> {
        vec![
            PhysicalDeviceProperties {
                device_name: "Simulated Discrete GPU".to_string(),
                device_type: PhysicalDeviceType::DiscreteGpu,
                vendor_id: 0x10DE,
                device_id: 0x2204,
                max_compute_work_group_count: [65535, 65535, 65535],
                max_compute_work_group_size: [1024, 1024, 64],
                max_push_constant_size: 256,
            },
            PhysicalDeviceProperties {
                device_name: "Simulated Integrated GPU".to_string(),
                device_type: PhysicalDeviceType::IntegratedGpu,
                vendor_id: 0x8086,
                device_id: 0x9A49,
                max_compute_work_group_count: [65535, 65535, 65535],
                max_compute_work_group_size: [512, 512, 64],
                max_push_constant_size: 128,
            },
        ]
    }

    /// Create a logical device from the given config.
    pub fn new(config: &VulkanConfig) -> Result<Self> {
        let devices = Self::enumerate_physical_devices();
        let physical = devices
            .into_iter()
            .nth(config.device_index as usize)
            .ok_or(VulkanError::NoSuitableDevice)?;

        let queue_families = vec![
            QueueFamilyProperties {
                index: 0,
                queue_count: 4,
                supports_compute: true,
                supports_transfer: true,
                supports_graphics: true,
            },
            QueueFamilyProperties {
                index: 1,
                queue_count: 2,
                supports_compute: true,
                supports_transfer: true,
                supports_graphics: false,
            },
            QueueFamilyProperties {
                index: 2,
                queue_count: 1,
                supports_compute: false,
                supports_transfer: true,
                supports_graphics: false,
            },
        ];

        let selected = if let Some(idx) = config.queue_family_index {
            let qf = queue_families
                .iter()
                .find(|q| q.index == idx && q.supports_compute)
                .ok_or(VulkanError::NoComputeQueueFamily)?;
            qf.index
        } else {
            // Prefer a dedicated compute queue (no graphics).
            queue_families
                .iter()
                .find(|q| q.supports_compute && !q.supports_graphics)
                .or_else(|| queue_families.iter().find(|q| q.supports_compute))
                .ok_or(VulkanError::NoComputeQueueFamily)?
                .index
        };

        Ok(Self {
            max_push_constant_size: physical.max_push_constant_size,
            physical_properties: physical,
            queue_families,
            selected_queue_family: selected,
        })
    }

    /// Create a device with no compute queues (for testing error paths).
    pub const fn new_without_compute() -> Result<Self> {
        Err(VulkanError::NoComputeQueueFamily)
    }

    pub const fn physical_properties(&self) -> &PhysicalDeviceProperties {
        &self.physical_properties
    }

    pub fn queue_families(&self) -> &[QueueFamilyProperties] {
        &self.queue_families
    }

    pub const fn selected_queue_family(&self) -> u32 {
        self.selected_queue_family
    }

    pub const fn max_push_constant_size(&self) -> usize {
        self.max_push_constant_size
    }
}

// ── Buffers ─────────────────────────────────────────────────────────────────

/// Buffer memory type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferMemoryType {
    /// GPU-only memory, fastest for compute.
    DeviceLocal,
    /// CPU-visible, coherent mapping.
    HostVisible,
    /// Staging buffer: host-visible, used for transfers.
    Staging,
}

/// GPU buffer backed by a CPU `Vec<u8>` for simulation.
#[derive(Debug)]
pub struct VulkanBuffer {
    memory_type: BufferMemoryType,
    size: usize,
    data: Vec<u8>,
    mapped: bool,
}

impl VulkanBuffer {
    /// Allocate a buffer of the given size and memory type.
    pub fn new(size: usize, memory_type: BufferMemoryType) -> Result<Self> {
        if size == 0 {
            return Err(VulkanError::BufferAllocationFailed("zero-size buffer".to_string()));
        }
        Ok(Self { memory_type, size, data: vec![0u8; size], mapped: false })
    }

    pub const fn memory_type(&self) -> BufferMemoryType {
        self.memory_type
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    /// Map the buffer for host access (only valid for host-visible/staging).
    pub fn map(&mut self) -> Result<&mut [u8]> {
        if self.memory_type == BufferMemoryType::DeviceLocal {
            Err(VulkanError::BufferAllocationFailed("cannot map device-local buffer".to_string()))
        } else {
            self.mapped = true;
            Ok(&mut self.data)
        }
    }

    /// Unmap the buffer.
    pub const fn unmap(&mut self) {
        self.mapped = false;
    }

    pub const fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Write `src` into the buffer at `offset`.
    pub fn write(&mut self, offset: usize, src: &[u8]) -> Result<()> {
        if offset + src.len() > self.size {
            return Err(VulkanError::BufferAllocationFailed("write out of bounds".to_string()));
        }
        self.data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Read from the buffer at `offset`.
    pub fn read(&self, offset: usize, len: usize) -> Result<&[u8]> {
        if offset + len > self.size {
            return Err(VulkanError::BufferAllocationFailed("read out of bounds".to_string()));
        }
        Ok(&self.data[offset..offset + len])
    }

    /// Simulated staging transfer from `src` into `self`.
    pub fn copy_from(
        &mut self,
        src: &Self,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let src_data = src.read(src_offset, len).map_err(|_| {
            VulkanError::BufferAllocationFailed("staging copy: source read failed".to_string())
        })?;
        let src_copy = src_data.to_vec();
        self.write(dst_offset, &src_copy)
    }
}

// ── Descriptor Sets ─────────────────────────────────────────────────────────

/// Type of descriptor binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorType {
    StorageBuffer,
    UniformBuffer,
}

/// A single binding within a descriptor set layout.
#[derive(Debug, Clone)]
pub struct DescriptorBinding {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub count: u32,
}

/// Descriptor set layout + allocation for shader bindings.
#[derive(Debug)]
pub struct VulkanDescriptorSet {
    bindings: Vec<DescriptorBinding>,
    bound_buffers: HashMap<u32, usize>,
}

impl VulkanDescriptorSet {
    /// Create a descriptor set with the specified layout bindings.
    pub fn new(bindings: Vec<DescriptorBinding>) -> Self {
        Self { bindings, bound_buffers: HashMap::new() }
    }

    /// Bind a buffer (by index) to the given binding point.
    pub fn bind_buffer(&mut self, binding: u32, buffer_index: usize) -> Result<()> {
        let max_binding = self.bindings.iter().map(|b| b.binding).max().unwrap_or(0);
        if !self.bindings.iter().any(|b| b.binding == binding) {
            return Err(VulkanError::DescriptorBindingOutOfRange { binding, max: max_binding });
        }
        self.bound_buffers.insert(binding, buffer_index);
        Ok(())
    }

    /// Whether all bindings have a buffer bound.
    pub fn is_fully_bound(&self) -> bool {
        self.bindings.iter().all(|b| self.bound_buffers.contains_key(&b.binding))
    }

    pub fn bindings(&self) -> &[DescriptorBinding] {
        &self.bindings
    }

    pub const fn bound_buffers(&self) -> &HashMap<u32, usize> {
        &self.bound_buffers
    }

    /// Number of bindings in the layout.
    pub const fn binding_count(&self) -> usize {
        self.bindings.len()
    }
}

// ── Shader Module ───────────────────────────────────────────────────────────

/// Wraps loaded SPIR-V bytecode as a shader module.
#[derive(Debug)]
pub struct VulkanShaderModule {
    spirv: Vec<u8>,
    entry_point: String,
}

impl VulkanShaderModule {
    /// Create a shader module from SPIR-V bytes.
    ///
    /// The SPIR-V magic number `0x07230203` is validated.
    pub fn new(spirv: &[u8], entry_point: &str) -> Result<Self> {
        if spirv.len() < 4 {
            return Err(VulkanError::InvalidShaderModule("SPIR-V too short".to_string()));
        }
        // Validate SPIR-V magic number (little-endian).
        let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
        if magic != 0x0723_0203 {
            return Err(VulkanError::InvalidShaderModule(format!(
                "bad SPIR-V magic: {magic:#010x}"
            )));
        }
        if entry_point.is_empty() {
            return Err(VulkanError::InvalidShaderModule(
                "entry point must not be empty".to_string(),
            ));
        }
        Ok(Self { spirv: spirv.to_vec(), entry_point: entry_point.to_string() })
    }

    pub fn spirv(&self) -> &[u8] {
        &self.spirv
    }

    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    pub const fn spirv_size(&self) -> usize {
        self.spirv.len()
    }
}

// ── Compute Pipeline ────────────────────────────────────────────────────────

/// Unique pipeline identifier for caching.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub shader_hash: u64,
    pub entry_point: String,
}

/// Pipeline layout (push constant ranges) + compute pipeline.
#[derive(Debug)]
pub struct VulkanComputePipeline {
    key: PipelineKey,
    push_constant_size: usize,
    created: bool,
}

impl VulkanComputePipeline {
    /// Create a compute pipeline from a shader module.
    pub fn new(
        shader: &VulkanShaderModule,
        push_constant_size: usize,
        max_push_constant_size: usize,
    ) -> Result<Self> {
        if push_constant_size > max_push_constant_size {
            return Err(VulkanError::PushConstantOverflow {
                actual: push_constant_size,
                limit: max_push_constant_size,
            });
        }
        let shader_hash = simple_hash(shader.spirv());
        Ok(Self {
            key: PipelineKey { shader_hash, entry_point: shader.entry_point().to_string() },
            push_constant_size,
            created: true,
        })
    }

    pub const fn key(&self) -> &PipelineKey {
        &self.key
    }

    pub const fn push_constant_size(&self) -> usize {
        self.push_constant_size
    }

    pub const fn is_created(&self) -> bool {
        self.created
    }
}

/// Simple non-cryptographic hash for pipeline caching.
fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

// ── Pipeline Cache ──────────────────────────────────────────────────────────

/// Simple pipeline cache keyed by `PipelineKey`.
#[derive(Debug, Default)]
pub struct PipelineCache {
    entries: HashSet<PipelineKey>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a pipeline into the cache. Returns `true` if newly inserted.
    pub fn insert(&mut self, key: PipelineKey) -> bool {
        self.entries.insert(key)
    }

    pub fn contains(&self, key: &PipelineKey) -> bool {
        self.entries.contains(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Command Buffer ──────────────────────────────────────────────────────────

/// Memory barrier scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierScope {
    /// Compute shader read/write.
    ComputeShader,
    /// Transfer operations.
    Transfer,
    /// Host access.
    Host,
}

/// Recorded commands within a command buffer.
#[derive(Debug, Clone)]
pub enum RecordedCommand {
    /// Dispatch compute work groups.
    Dispatch { group_count_x: u32, group_count_y: u32, group_count_z: u32 },
    /// Copy between buffers.
    CopyBuffer {
        src_index: usize,
        dst_index: usize,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    },
    /// Pipeline barrier.
    Barrier { src_scope: BarrierScope, dst_scope: BarrierScope },
    /// Push constants update.
    PushConstants { offset: usize, data: Vec<u8> },
    /// Bind a descriptor set.
    BindDescriptorSet { set_index: usize },
    /// Bind a pipeline.
    BindPipeline { pipeline_index: usize },
}

/// Command buffer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferState {
    Initial,
    Recording,
    Executable,
    Submitted,
}

/// Records dispatch commands, buffer copies, and barriers.
#[derive(Debug)]
pub struct VulkanCommandBuffer {
    state: CommandBufferState,
    commands: Vec<RecordedCommand>,
}

impl VulkanCommandBuffer {
    pub const fn new() -> Self {
        Self { state: CommandBufferState::Initial, commands: Vec::new() }
    }

    /// Begin recording commands.
    pub fn begin(&mut self) -> Result<()> {
        if self.state != CommandBufferState::Initial && self.state != CommandBufferState::Executable
        {
            return Err(VulkanError::CommandBufferError("cannot begin: invalid state".to_string()));
        }
        self.state = CommandBufferState::Recording;
        self.commands.clear();
        Ok(())
    }

    /// End recording.
    pub fn end(&mut self) -> Result<()> {
        if self.state != CommandBufferState::Recording {
            return Err(VulkanError::CommandBufferError("cannot end: not recording".to_string()));
        }
        self.state = CommandBufferState::Executable;
        Ok(())
    }

    /// Record a compute dispatch.
    pub fn cmd_dispatch(&mut self, x: u32, y: u32, z: u32) -> Result<()> {
        self.ensure_recording()?;
        if x == 0 || y == 0 || z == 0 {
            return Err(VulkanError::ZeroWorkGroups);
        }
        self.commands.push(RecordedCommand::Dispatch {
            group_count_x: x,
            group_count_y: y,
            group_count_z: z,
        });
        Ok(())
    }

    /// Record a buffer copy.
    pub fn cmd_copy_buffer(
        &mut self,
        src_index: usize,
        dst_index: usize,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        self.ensure_recording()?;
        self.commands.push(RecordedCommand::CopyBuffer {
            src_index,
            dst_index,
            src_offset,
            dst_offset,
            size,
        });
        Ok(())
    }

    /// Record a pipeline barrier.
    pub fn cmd_barrier(&mut self, src: BarrierScope, dst: BarrierScope) -> Result<()> {
        self.ensure_recording()?;
        self.commands.push(RecordedCommand::Barrier { src_scope: src, dst_scope: dst });
        Ok(())
    }

    /// Record a push constants update.
    pub fn cmd_push_constants(
        &mut self,
        offset: usize,
        data: &[u8],
        max_size: usize,
    ) -> Result<()> {
        self.ensure_recording()?;
        if offset + data.len() > max_size {
            return Err(VulkanError::PushConstantOverflow {
                actual: offset + data.len(),
                limit: max_size,
            });
        }
        self.commands.push(RecordedCommand::PushConstants { offset, data: data.to_vec() });
        Ok(())
    }

    /// Record binding a descriptor set.
    pub fn cmd_bind_descriptor_set(&mut self, set_index: usize) -> Result<()> {
        self.ensure_recording()?;
        self.commands.push(RecordedCommand::BindDescriptorSet { set_index });
        Ok(())
    }

    /// Record binding a compute pipeline.
    pub fn cmd_bind_pipeline(&mut self, pipeline_index: usize) -> Result<()> {
        self.ensure_recording()?;
        self.commands.push(RecordedCommand::BindPipeline { pipeline_index });
        Ok(())
    }

    pub const fn state(&self) -> CommandBufferState {
        self.state
    }

    pub fn commands(&self) -> &[RecordedCommand] {
        &self.commands
    }

    pub const fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Mark as submitted (called by queue submit).
    pub const fn mark_submitted(&mut self) {
        self.state = CommandBufferState::Submitted;
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.state = CommandBufferState::Initial;
        self.commands.clear();
    }

    fn ensure_recording(&self) -> Result<()> {
        if self.state != CommandBufferState::Recording {
            return Err(VulkanError::CommandBufferError("not recording".to_string()));
        }
        Ok(())
    }
}

impl Default for VulkanCommandBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fence ───────────────────────────────────────────────────────────────────

/// Synchronisation fence for host-device coordination.
#[derive(Debug)]
pub struct VulkanFence {
    signaled: Arc<AtomicBool>,
    wait_count: AtomicU64,
}

impl VulkanFence {
    /// Create an unsignaled fence.
    pub fn new() -> Self {
        Self { signaled: Arc::new(AtomicBool::new(false)), wait_count: AtomicU64::new(0) }
    }

    /// Create a pre-signaled fence.
    pub fn new_signaled() -> Self {
        Self { signaled: Arc::new(AtomicBool::new(true)), wait_count: AtomicU64::new(0) }
    }

    /// Signal the fence (simulates GPU work completion).
    pub fn signal(&self) {
        self.signaled.store(true, Ordering::Release);
    }

    /// Reset the fence to unsignaled.
    pub fn reset(&self) {
        self.signaled.store(false, Ordering::Release);
    }

    /// Check if signaled without blocking.
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::Acquire)
    }

    /// Wait for the fence with a timeout.
    pub fn wait(&self, timeout: Duration) -> Result<()> {
        self.wait_count.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();
        while !self.signaled.load(Ordering::Acquire) {
            if start.elapsed() >= timeout {
                return Err(VulkanError::FenceTimeout);
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Number of times `wait` has been called.
    pub fn wait_count(&self) -> u64 {
        self.wait_count.load(Ordering::Relaxed)
    }
}

impl Default for VulkanFence {
    fn default() -> Self {
        Self::new()
    }
}

// ── Compute Engine ──────────────────────────────────────────────────────────

/// Orchestrator: instance → device → pipeline → bind → dispatch → sync.
#[derive(Debug)]
pub struct VulkanComputeEngine {
    instance: VulkanInstance,
    device: VulkanDevice,
    buffers: Vec<VulkanBuffer>,
    descriptor_sets: Vec<VulkanDescriptorSet>,
    pipelines: Vec<VulkanComputePipeline>,
    pipeline_cache: PipelineCache,
    command_buffers: Vec<VulkanCommandBuffer>,
    fences: Vec<VulkanFence>,
    submissions: u64,
}

impl VulkanComputeEngine {
    /// Create a new compute engine from config.
    pub fn new(config: &VulkanConfig) -> Result<Self> {
        let instance = VulkanInstance::new(config)?;
        let device = VulkanDevice::new(config)?;
        Ok(Self {
            instance,
            device,
            buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            pipelines: Vec::new(),
            pipeline_cache: PipelineCache::new(),
            command_buffers: Vec::new(),
            fences: Vec::new(),
            submissions: 0,
        })
    }

    pub const fn instance(&self) -> &VulkanInstance {
        &self.instance
    }

    pub const fn instance_mut(&mut self) -> &mut VulkanInstance {
        &mut self.instance
    }

    pub const fn device(&self) -> &VulkanDevice {
        &self.device
    }

    // ── Buffer management ───────────────────────────────────────────────

    /// Allocate a buffer and return its index.
    pub fn create_buffer(&mut self, size: usize, memory_type: BufferMemoryType) -> Result<usize> {
        let buf = VulkanBuffer::new(size, memory_type)?;
        let idx = self.buffers.len();
        self.buffers.push(buf);
        Ok(idx)
    }

    pub fn buffer(&self, index: usize) -> Option<&VulkanBuffer> {
        self.buffers.get(index)
    }

    pub fn buffer_mut(&mut self, index: usize) -> Option<&mut VulkanBuffer> {
        self.buffers.get_mut(index)
    }

    pub const fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Perform a staging copy between two engine-owned buffers.
    pub fn copy_buffer(
        &mut self,
        src_idx: usize,
        dst_idx: usize,
        src_offset: usize,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        if src_idx == dst_idx {
            return Err(VulkanError::BufferAllocationFailed(
                "cannot copy buffer to itself".to_string(),
            ));
        }
        let (left, right) = if src_idx < dst_idx {
            let (a, b) = self.buffers.split_at_mut(dst_idx);
            (&a[src_idx], &mut b[0])
        } else {
            let (a, b) = self.buffers.split_at_mut(src_idx);
            let dst = &mut a[dst_idx];
            let src = &b[0];
            (src as &VulkanBuffer, dst)
        };
        right.copy_from(left, src_offset, dst_offset, len)
    }

    // ── Descriptor sets ─────────────────────────────────────────────────

    /// Create a descriptor set and return its index.
    pub fn create_descriptor_set(&mut self, bindings: Vec<DescriptorBinding>) -> usize {
        let ds = VulkanDescriptorSet::new(bindings);
        let idx = self.descriptor_sets.len();
        self.descriptor_sets.push(ds);
        idx
    }

    pub fn descriptor_set(&self, index: usize) -> Option<&VulkanDescriptorSet> {
        self.descriptor_sets.get(index)
    }

    pub fn descriptor_set_mut(&mut self, index: usize) -> Option<&mut VulkanDescriptorSet> {
        self.descriptor_sets.get_mut(index)
    }

    // ── Pipeline management ─────────────────────────────────────────────

    /// Load a shader, create a pipeline, cache it, and return its index.
    pub fn create_pipeline(
        &mut self,
        spirv: &[u8],
        entry_point: &str,
        push_constant_size: usize,
    ) -> Result<usize> {
        let shader = VulkanShaderModule::new(spirv, entry_point)?;
        let pipeline = VulkanComputePipeline::new(
            &shader,
            push_constant_size,
            self.device.max_push_constant_size(),
        )?;
        self.pipeline_cache.insert(pipeline.key().clone());
        let idx = self.pipelines.len();
        self.pipelines.push(pipeline);
        Ok(idx)
    }

    pub fn pipeline(&self, index: usize) -> Option<&VulkanComputePipeline> {
        self.pipelines.get(index)
    }

    pub const fn pipeline_cache(&self) -> &PipelineCache {
        &self.pipeline_cache
    }

    // ── Command buffers ─────────────────────────────────────────────────

    /// Allocate a command buffer and return its index.
    pub fn create_command_buffer(&mut self) -> usize {
        let idx = self.command_buffers.len();
        self.command_buffers.push(VulkanCommandBuffer::new());
        idx
    }

    pub fn command_buffer(&self, index: usize) -> Option<&VulkanCommandBuffer> {
        self.command_buffers.get(index)
    }

    pub fn command_buffer_mut(&mut self, index: usize) -> Option<&mut VulkanCommandBuffer> {
        self.command_buffers.get_mut(index)
    }

    // ── Fences ──────────────────────────────────────────────────────────

    /// Create a fence and return its index.
    pub fn create_fence(&mut self, signaled: bool) -> usize {
        let fence = if signaled { VulkanFence::new_signaled() } else { VulkanFence::new() };
        let idx = self.fences.len();
        self.fences.push(fence);
        idx
    }

    pub fn fence(&self, index: usize) -> Option<&VulkanFence> {
        self.fences.get(index)
    }

    // ── Submission ──────────────────────────────────────────────────────

    /// Submit an executable command buffer, signal the fence on "completion".
    pub fn submit(&mut self, cmd_index: usize, fence_index: usize) -> Result<()> {
        let cb = self.command_buffers.get_mut(cmd_index).ok_or_else(|| {
            VulkanError::CommandBufferError("invalid command buffer index".to_string())
        })?;
        if cb.state() != CommandBufferState::Executable {
            return Err(VulkanError::CommandBufferError(
                "command buffer not executable".to_string(),
            ));
        }
        cb.mark_submitted();

        // Execute recorded buffer copies on CPU.
        let commands: Vec<RecordedCommand> = cb.commands().to_vec();
        for cmd in &commands {
            if let RecordedCommand::CopyBuffer {
                src_index,
                dst_index,
                src_offset,
                dst_offset,
                size,
            } = cmd
            {
                self.copy_buffer(*src_index, *dst_index, *src_offset, *dst_offset, *size)?;
            }
        }

        let fence = self
            .fences
            .get(fence_index)
            .ok_or_else(|| VulkanError::CommandBufferError("invalid fence index".to_string()))?;
        fence.signal();
        self.submissions += 1;
        Ok(())
    }

    /// Wait for a fence with timeout.
    pub fn wait_fence(&self, fence_index: usize, timeout: Duration) -> Result<()> {
        let fence = self
            .fences
            .get(fence_index)
            .ok_or_else(|| VulkanError::CommandBufferError("invalid fence index".to_string()))?;
        fence.wait(timeout)
    }

    pub const fn submission_count(&self) -> u64 {
        self.submissions
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: valid SPIR-V stub (magic + padding).
    fn spirv_stub() -> Vec<u8> {
        let mut v = 0x0723_0203u32.to_le_bytes().to_vec();
        v.extend_from_slice(&[0u8; 12]); // padding to 16 bytes
        v
    }

    fn default_config() -> VulkanConfig {
        VulkanConfig::default()
    }

    fn config_with_validation() -> VulkanConfig {
        VulkanConfig { enable_validation_layers: true, ..VulkanConfig::default() }
    }

    // ── Instance tests ──────────────────────────────────────────────────

    #[test]
    fn instance_creation_default() {
        let inst = VulkanInstance::new(&default_config()).unwrap();
        assert!(inst.is_created());
        assert!(!inst.validation_enabled());
    }

    #[test]
    fn instance_creation_with_validation() {
        let mut inst = VulkanInstance::new(&config_with_validation()).unwrap();
        assert!(inst.validation_enabled());
        let msgs = inst.drain_debug_messages();
        assert!(!msgs.is_empty());
        assert_eq!(msgs[0].severity, DebugSeverity::Info);
    }

    #[test]
    fn instance_empty_app_name_fails() {
        let cfg = VulkanConfig { application_name: String::new(), ..default_config() };
        let err = VulkanInstance::new(&cfg).unwrap_err();
        assert!(matches!(err, VulkanError::InstanceCreationFailed(_)));
    }

    #[test]
    fn instance_config_roundtrip() {
        let cfg = VulkanConfig {
            application_name: "test-app".to_string(),
            application_version: 42,
            ..default_config()
        };
        let inst = VulkanInstance::new(&cfg).unwrap();
        assert_eq!(inst.config().application_name, "test-app");
        assert_eq!(inst.config().application_version, 42);
    }

    #[test]
    fn instance_drain_empty_without_validation() {
        let mut inst = VulkanInstance::new(&default_config()).unwrap();
        assert!(inst.drain_debug_messages().is_empty());
    }

    #[test]
    fn instance_drain_clears_messages() {
        let mut inst = VulkanInstance::new(&config_with_validation()).unwrap();
        assert!(!inst.drain_debug_messages().is_empty());
        assert!(inst.drain_debug_messages().is_empty());
    }

    // ── Device tests ────────────────────────────────────────────────────

    #[test]
    fn enumerate_physical_devices() {
        let devs = VulkanDevice::enumerate_physical_devices();
        assert!(devs.len() >= 2);
        assert_eq!(devs[0].device_type, PhysicalDeviceType::DiscreteGpu);
        assert_eq!(devs[1].device_type, PhysicalDeviceType::IntegratedGpu);
    }

    #[test]
    fn device_creation_default() {
        let dev = VulkanDevice::new(&default_config()).unwrap();
        assert_eq!(dev.physical_properties().device_type, PhysicalDeviceType::DiscreteGpu);
    }

    #[test]
    fn device_creation_second_device() {
        let cfg = VulkanConfig { device_index: 1, ..default_config() };
        let dev = VulkanDevice::new(&cfg).unwrap();
        assert_eq!(dev.physical_properties().device_type, PhysicalDeviceType::IntegratedGpu);
    }

    #[test]
    fn device_creation_invalid_index() {
        let cfg = VulkanConfig { device_index: 99, ..default_config() };
        let err = VulkanDevice::new(&cfg).unwrap_err();
        assert_eq!(err, VulkanError::NoSuitableDevice);
    }

    #[test]
    fn device_auto_selects_dedicated_compute_queue() {
        let dev = VulkanDevice::new(&default_config()).unwrap();
        // Queue family 1 is compute-only (no graphics).
        assert_eq!(dev.selected_queue_family(), 1);
    }

    #[test]
    fn device_explicit_queue_family() {
        let cfg = VulkanConfig { queue_family_index: Some(0), ..default_config() };
        let dev = VulkanDevice::new(&cfg).unwrap();
        assert_eq!(dev.selected_queue_family(), 0);
    }

    #[test]
    fn device_explicit_non_compute_queue_fails() {
        let cfg = VulkanConfig {
            queue_family_index: Some(2), // transfer-only
            ..default_config()
        };
        let err = VulkanDevice::new(&cfg).unwrap_err();
        assert_eq!(err, VulkanError::NoComputeQueueFamily);
    }

    #[test]
    fn device_no_compute_queues() {
        let err = VulkanDevice::new_without_compute().unwrap_err();
        assert_eq!(err, VulkanError::NoComputeQueueFamily);
    }

    #[test]
    fn device_queue_families_populated() {
        let dev = VulkanDevice::new(&default_config()).unwrap();
        assert_eq!(dev.queue_families().len(), 3);
        assert!(dev.queue_families()[0].supports_graphics);
        assert!(!dev.queue_families()[1].supports_graphics);
    }

    #[test]
    fn device_max_push_constant_size() {
        let dev = VulkanDevice::new(&default_config()).unwrap();
        assert!(dev.max_push_constant_size() > 0);
    }

    #[test]
    fn physical_device_type_display() {
        assert_eq!(PhysicalDeviceType::DiscreteGpu.to_string(), "Discrete GPU");
        assert_eq!(PhysicalDeviceType::Cpu.to_string(), "CPU");
        assert_eq!(PhysicalDeviceType::Other.to_string(), "Other");
    }

    // ── Buffer tests ────────────────────────────────────────────────────

    #[test]
    fn buffer_device_local() {
        let buf = VulkanBuffer::new(1024, BufferMemoryType::DeviceLocal).unwrap();
        assert_eq!(buf.size(), 1024);
        assert_eq!(buf.memory_type(), BufferMemoryType::DeviceLocal);
    }

    #[test]
    fn buffer_host_visible() {
        let mut buf = VulkanBuffer::new(256, BufferMemoryType::HostVisible).unwrap();
        assert!(!buf.is_mapped());
        let slice = buf.map().unwrap();
        assert_eq!(slice.len(), 256);
        buf.unmap();
        assert!(!buf.is_mapped());
    }

    #[test]
    fn buffer_staging() {
        let buf = VulkanBuffer::new(512, BufferMemoryType::Staging).unwrap();
        assert_eq!(buf.memory_type(), BufferMemoryType::Staging);
    }

    #[test]
    fn buffer_zero_size_fails() {
        let err = VulkanBuffer::new(0, BufferMemoryType::DeviceLocal).unwrap_err();
        assert!(matches!(err, VulkanError::BufferAllocationFailed(_)));
    }

    #[test]
    fn buffer_device_local_map_fails() {
        let mut buf = VulkanBuffer::new(64, BufferMemoryType::DeviceLocal).unwrap();
        assert!(buf.map().is_err());
    }

    #[test]
    fn buffer_write_and_read() {
        let mut buf = VulkanBuffer::new(64, BufferMemoryType::HostVisible).unwrap();
        buf.write(0, &[1, 2, 3, 4]).unwrap();
        let data = buf.read(0, 4).unwrap();
        assert_eq!(data, &[1, 2, 3, 4]);
    }

    #[test]
    fn buffer_write_out_of_bounds() {
        let mut buf = VulkanBuffer::new(4, BufferMemoryType::HostVisible).unwrap();
        let err = buf.write(2, &[1, 2, 3]).unwrap_err();
        assert!(matches!(err, VulkanError::BufferAllocationFailed(_)));
    }

    #[test]
    fn buffer_read_out_of_bounds() {
        let buf = VulkanBuffer::new(4, BufferMemoryType::HostVisible).unwrap();
        let err = buf.read(2, 4).unwrap_err();
        assert!(matches!(err, VulkanError::BufferAllocationFailed(_)));
    }

    #[test]
    fn buffer_staging_transfer() {
        let mut src = VulkanBuffer::new(16, BufferMemoryType::Staging).unwrap();
        src.write(0, &[10, 20, 30, 40]).unwrap();
        let mut dst = VulkanBuffer::new(16, BufferMemoryType::DeviceLocal).unwrap();
        dst.copy_from(&src, 0, 4, 4).unwrap();
        let data = dst.read(4, 4).unwrap();
        assert_eq!(data, &[10, 20, 30, 40]);
    }

    #[test]
    fn buffer_write_at_offset() {
        let mut buf = VulkanBuffer::new(16, BufferMemoryType::HostVisible).unwrap();
        buf.write(8, &[0xAA, 0xBB]).unwrap();
        assert_eq!(buf.read(8, 2).unwrap(), &[0xAA, 0xBB]);
        assert_eq!(buf.read(0, 1).unwrap(), &[0x00]);
    }

    #[test]
    fn buffer_staging_map_works() {
        let mut buf = VulkanBuffer::new(8, BufferMemoryType::Staging).unwrap();
        let slice = buf.map().unwrap();
        slice[0] = 42;
        assert_eq!(buf.read(0, 1).unwrap(), &[42]);
    }

    // ── Descriptor set tests ────────────────────────────────────────────

    #[test]
    fn descriptor_set_creation() {
        let ds = VulkanDescriptorSet::new(vec![
            DescriptorBinding {
                binding: 0,
                descriptor_type: DescriptorType::StorageBuffer,
                count: 1,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: DescriptorType::UniformBuffer,
                count: 1,
            },
        ]);
        assert_eq!(ds.binding_count(), 2);
        assert!(!ds.is_fully_bound());
    }

    #[test]
    fn descriptor_set_bind_buffer() {
        let mut ds = VulkanDescriptorSet::new(vec![DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        }]);
        ds.bind_buffer(0, 0).unwrap();
        assert!(ds.is_fully_bound());
    }

    #[test]
    fn descriptor_set_bind_invalid_binding() {
        let mut ds = VulkanDescriptorSet::new(vec![DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        }]);
        let err = ds.bind_buffer(5, 0).unwrap_err();
        assert!(matches!(err, VulkanError::DescriptorBindingOutOfRange { .. }));
    }

    #[test]
    fn descriptor_set_partial_binding() {
        let mut ds = VulkanDescriptorSet::new(vec![
            DescriptorBinding {
                binding: 0,
                descriptor_type: DescriptorType::StorageBuffer,
                count: 1,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: DescriptorType::StorageBuffer,
                count: 1,
            },
        ]);
        ds.bind_buffer(0, 0).unwrap();
        assert!(!ds.is_fully_bound());
        ds.bind_buffer(1, 1).unwrap();
        assert!(ds.is_fully_bound());
    }

    #[test]
    fn descriptor_set_bound_buffers_map() {
        let mut ds = VulkanDescriptorSet::new(vec![DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        }]);
        ds.bind_buffer(0, 42).unwrap();
        assert_eq!(ds.bound_buffers().get(&0), Some(&42));
    }

    #[test]
    fn descriptor_set_rebind_same_binding() {
        let mut ds = VulkanDescriptorSet::new(vec![DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        }]);
        ds.bind_buffer(0, 1).unwrap();
        ds.bind_buffer(0, 2).unwrap();
        assert_eq!(ds.bound_buffers().get(&0), Some(&2));
    }

    #[test]
    fn descriptor_set_empty_bindings() {
        let ds = VulkanDescriptorSet::new(vec![]);
        assert_eq!(ds.binding_count(), 0);
        assert!(ds.is_fully_bound()); // vacuously true
    }

    // ── Shader module tests ─────────────────────────────────────────────

    #[test]
    fn shader_module_valid() {
        let shader = VulkanShaderModule::new(&spirv_stub(), "main").unwrap();
        assert_eq!(shader.entry_point(), "main");
        assert_eq!(shader.spirv_size(), 16);
    }

    #[test]
    fn shader_module_empty_spirv() {
        let err = VulkanShaderModule::new(&[], "main").unwrap_err();
        assert!(matches!(err, VulkanError::InvalidShaderModule(_)));
    }

    #[test]
    fn shader_module_short_spirv() {
        let err = VulkanShaderModule::new(&[0, 1, 2], "main").unwrap_err();
        assert!(matches!(err, VulkanError::InvalidShaderModule(_)));
    }

    #[test]
    fn shader_module_bad_magic() {
        let bad = [0u8; 16];
        let err = VulkanShaderModule::new(&bad, "main").unwrap_err();
        assert!(matches!(err, VulkanError::InvalidShaderModule(_)));
    }

    #[test]
    fn shader_module_empty_entry_point() {
        let err = VulkanShaderModule::new(&spirv_stub(), "").unwrap_err();
        assert!(matches!(err, VulkanError::InvalidShaderModule(_)));
    }

    #[test]
    fn shader_module_spirv_data_preserved() {
        let spirv = spirv_stub();
        let shader = VulkanShaderModule::new(&spirv, "main").unwrap();
        assert_eq!(shader.spirv(), spirv.as_slice());
    }

    #[test]
    fn shader_module_custom_entry_point() {
        let shader = VulkanShaderModule::new(&spirv_stub(), "compute_kernel").unwrap();
        assert_eq!(shader.entry_point(), "compute_kernel");
    }

    // ── Pipeline tests ──────────────────────────────────────────────────

    #[test]
    fn pipeline_creation() {
        let shader = VulkanShaderModule::new(&spirv_stub(), "main").unwrap();
        let pipe = VulkanComputePipeline::new(&shader, 64, 256).unwrap();
        assert!(pipe.is_created());
        assert_eq!(pipe.push_constant_size(), 64);
    }

    #[test]
    fn pipeline_push_constant_overflow() {
        let shader = VulkanShaderModule::new(&spirv_stub(), "main").unwrap();
        let err = VulkanComputePipeline::new(&shader, 300, 256).unwrap_err();
        assert!(matches!(err, VulkanError::PushConstantOverflow { actual: 300, limit: 256 }));
    }

    #[test]
    fn pipeline_key_stable() {
        let shader = VulkanShaderModule::new(&spirv_stub(), "main").unwrap();
        let p1 = VulkanComputePipeline::new(&shader, 0, 256).unwrap();
        let p2 = VulkanComputePipeline::new(&shader, 0, 256).unwrap();
        assert_eq!(p1.key(), p2.key());
    }

    #[test]
    fn pipeline_different_entry_points_different_keys() {
        let s1 = VulkanShaderModule::new(&spirv_stub(), "main").unwrap();
        let s2 = VulkanShaderModule::new(&spirv_stub(), "alt").unwrap();
        let p1 = VulkanComputePipeline::new(&s1, 0, 256).unwrap();
        let p2 = VulkanComputePipeline::new(&s2, 0, 256).unwrap();
        assert_ne!(p1.key(), p2.key());
    }

    #[test]
    fn pipeline_zero_push_constants() {
        let shader = VulkanShaderModule::new(&spirv_stub(), "main").unwrap();
        let pipe = VulkanComputePipeline::new(&shader, 0, 256).unwrap();
        assert_eq!(pipe.push_constant_size(), 0);
    }

    // ── Pipeline cache tests ────────────────────────────────────────────

    #[test]
    fn pipeline_cache_insert_and_lookup() {
        let mut cache = PipelineCache::new();
        assert!(cache.is_empty());
        let key = PipelineKey { shader_hash: 123, entry_point: "main".to_string() };
        assert!(cache.insert(key.clone()));
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn pipeline_cache_duplicate_insert() {
        let mut cache = PipelineCache::new();
        let key = PipelineKey { shader_hash: 123, entry_point: "main".to_string() };
        assert!(cache.insert(key.clone()));
        assert!(!cache.insert(key));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn pipeline_cache_multiple_entries() {
        let mut cache = PipelineCache::new();
        for i in 0..5 {
            cache.insert(PipelineKey { shader_hash: i, entry_point: "main".to_string() });
        }
        assert_eq!(cache.len(), 5);
    }

    // ── Command buffer tests ────────────────────────────────────────────

    #[test]
    fn command_buffer_initial_state() {
        let cb = VulkanCommandBuffer::new();
        assert_eq!(cb.state(), CommandBufferState::Initial);
        assert_eq!(cb.command_count(), 0);
    }

    #[test]
    fn command_buffer_begin_end() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        assert_eq!(cb.state(), CommandBufferState::Recording);
        cb.end().unwrap();
        assert_eq!(cb.state(), CommandBufferState::Executable);
    }

    #[test]
    fn command_buffer_record_dispatch() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_dispatch(8, 4, 1).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_dispatch_zero_x() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        let err = cb.cmd_dispatch(0, 1, 1).unwrap_err();
        assert_eq!(err, VulkanError::ZeroWorkGroups);
    }

    #[test]
    fn command_buffer_dispatch_zero_y() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        let err = cb.cmd_dispatch(1, 0, 1).unwrap_err();
        assert_eq!(err, VulkanError::ZeroWorkGroups);
    }

    #[test]
    fn command_buffer_dispatch_zero_z() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        let err = cb.cmd_dispatch(1, 1, 0).unwrap_err();
        assert_eq!(err, VulkanError::ZeroWorkGroups);
    }

    #[test]
    fn command_buffer_record_barrier() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_barrier(BarrierScope::ComputeShader, BarrierScope::Transfer).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_record_copy() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_copy_buffer(0, 1, 0, 0, 256).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_push_constants() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_push_constants(0, &[1, 2, 3, 4], 128).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_push_constants_overflow() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        let data = vec![0u8; 64];
        let err = cb.cmd_push_constants(100, &data, 128).unwrap_err();
        assert!(matches!(err, VulkanError::PushConstantOverflow { .. }));
    }

    #[test]
    fn command_buffer_bind_descriptor_set() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_bind_descriptor_set(0).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_bind_pipeline() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_bind_pipeline(0).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_cannot_record_without_begin() {
        let mut cb = VulkanCommandBuffer::new();
        let err = cb.cmd_dispatch(1, 1, 1).unwrap_err();
        assert!(matches!(err, VulkanError::CommandBufferError(_)));
    }

    #[test]
    fn command_buffer_cannot_end_without_begin() {
        let mut cb = VulkanCommandBuffer::new();
        assert!(cb.end().is_err());
    }

    #[test]
    fn command_buffer_reset() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_dispatch(1, 1, 1).unwrap();
        cb.end().unwrap();
        cb.reset();
        assert_eq!(cb.state(), CommandBufferState::Initial);
        assert_eq!(cb.command_count(), 0);
    }

    #[test]
    fn command_buffer_re_record_after_executable() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.end().unwrap();
        // Can begin again from Executable state.
        cb.begin().unwrap();
        cb.cmd_dispatch(2, 2, 2).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 1);
    }

    #[test]
    fn command_buffer_multiple_commands() {
        let mut cb = VulkanCommandBuffer::new();
        cb.begin().unwrap();
        cb.cmd_bind_pipeline(0).unwrap();
        cb.cmd_bind_descriptor_set(0).unwrap();
        cb.cmd_push_constants(0, &[1, 2], 128).unwrap();
        cb.cmd_barrier(BarrierScope::ComputeShader, BarrierScope::ComputeShader).unwrap();
        cb.cmd_dispatch(4, 4, 1).unwrap();
        cb.cmd_barrier(BarrierScope::ComputeShader, BarrierScope::Host).unwrap();
        cb.end().unwrap();
        assert_eq!(cb.command_count(), 6);
    }

    #[test]
    fn command_buffer_default_trait() {
        let cb = VulkanCommandBuffer::default();
        assert_eq!(cb.state(), CommandBufferState::Initial);
    }

    // ── Fence tests ─────────────────────────────────────────────────────

    #[test]
    fn fence_unsignaled() {
        let fence = VulkanFence::new();
        assert!(!fence.is_signaled());
    }

    #[test]
    fn fence_pre_signaled() {
        let fence = VulkanFence::new_signaled();
        assert!(fence.is_signaled());
    }

    #[test]
    fn fence_signal_and_reset() {
        let fence = VulkanFence::new();
        fence.signal();
        assert!(fence.is_signaled());
        fence.reset();
        assert!(!fence.is_signaled());
    }

    #[test]
    fn fence_wait_signaled_immediate() {
        let fence = VulkanFence::new_signaled();
        fence.wait(Duration::from_millis(10)).unwrap();
    }

    #[test]
    fn fence_wait_timeout() {
        let fence = VulkanFence::new();
        let err = fence.wait(Duration::from_millis(1)).unwrap_err();
        assert_eq!(err, VulkanError::FenceTimeout);
    }

    #[test]
    fn fence_wait_count_increments() {
        let fence = VulkanFence::new_signaled();
        fence.wait(Duration::from_millis(10)).unwrap();
        fence.wait(Duration::from_millis(10)).unwrap();
        assert_eq!(fence.wait_count(), 2);
    }

    #[test]
    fn fence_default_unsignaled() {
        let fence = VulkanFence::default();
        assert!(!fence.is_signaled());
    }

    // ── Compute engine tests ────────────────────────────────────────────

    #[test]
    fn engine_creation() {
        let engine = VulkanComputeEngine::new(&default_config()).unwrap();
        assert!(engine.instance().is_created());
        assert_eq!(engine.submission_count(), 0);
    }

    #[test]
    fn engine_with_validation() {
        let engine = VulkanComputeEngine::new(&config_with_validation()).unwrap();
        assert!(engine.instance().validation_enabled());
    }

    #[test]
    fn engine_buffer_management() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let b0 = engine.create_buffer(256, BufferMemoryType::DeviceLocal).unwrap();
        let b1 = engine.create_buffer(128, BufferMemoryType::HostVisible).unwrap();
        assert_eq!(b0, 0);
        assert_eq!(b1, 1);
        assert_eq!(engine.buffer_count(), 2);
        assert_eq!(engine.buffer(0).unwrap().size(), 256);
    }

    #[test]
    fn engine_buffer_zero_size_fails() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        assert!(engine.create_buffer(0, BufferMemoryType::DeviceLocal).is_err());
    }

    #[test]
    fn engine_staging_copy() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let src = engine.create_buffer(64, BufferMemoryType::Staging).unwrap();
        let dst = engine.create_buffer(64, BufferMemoryType::DeviceLocal).unwrap();
        engine.buffer_mut(src).unwrap().write(0, &[1, 2, 3, 4]).unwrap();
        engine.copy_buffer(src, dst, 0, 0, 4).unwrap();
        let data = engine.buffer(dst).unwrap().read(0, 4).unwrap();
        assert_eq!(data, &[1, 2, 3, 4]);
    }

    #[test]
    fn engine_copy_buffer_to_self_fails() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let b = engine.create_buffer(64, BufferMemoryType::HostVisible).unwrap();
        assert!(engine.copy_buffer(b, b, 0, 0, 4).is_err());
    }

    #[test]
    fn engine_descriptor_set() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let ds_idx = engine.create_descriptor_set(vec![DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        }]);
        assert_eq!(ds_idx, 0);
        engine.descriptor_set_mut(ds_idx).unwrap().bind_buffer(0, 0).unwrap();
        assert!(engine.descriptor_set(ds_idx).unwrap().is_fully_bound());
    }

    #[test]
    fn engine_pipeline_creation() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let idx = engine.create_pipeline(&spirv_stub(), "main", 64).unwrap();
        assert_eq!(idx, 0);
        assert!(engine.pipeline(idx).unwrap().is_created());
        assert_eq!(engine.pipeline_cache().len(), 1);
    }

    #[test]
    fn engine_pipeline_push_constant_overflow() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let err = engine.create_pipeline(&spirv_stub(), "main", 9999).unwrap_err();
        assert!(matches!(err, VulkanError::PushConstantOverflow { .. }));
    }

    #[test]
    fn engine_full_dispatch_cycle() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();

        // Create buffers.
        let src = engine.create_buffer(64, BufferMemoryType::Staging).unwrap();
        let dst = engine.create_buffer(64, BufferMemoryType::DeviceLocal).unwrap();

        // Write source data.
        engine.buffer_mut(src).unwrap().write(0, &[10, 20, 30, 40]).unwrap();

        // Create descriptor set.
        let ds = engine.create_descriptor_set(vec![
            DescriptorBinding {
                binding: 0,
                descriptor_type: DescriptorType::StorageBuffer,
                count: 1,
            },
            DescriptorBinding {
                binding: 1,
                descriptor_type: DescriptorType::StorageBuffer,
                count: 1,
            },
        ]);
        engine.descriptor_set_mut(ds).unwrap().bind_buffer(0, src).unwrap();
        engine.descriptor_set_mut(ds).unwrap().bind_buffer(1, dst).unwrap();

        // Create pipeline.
        let pipe = engine.create_pipeline(&spirv_stub(), "main", 16).unwrap();

        // Record command buffer.
        let cb = engine.create_command_buffer();
        let cmds = engine.command_buffer_mut(cb).unwrap();
        cmds.begin().unwrap();
        cmds.cmd_bind_pipeline(pipe).unwrap();
        cmds.cmd_bind_descriptor_set(ds).unwrap();
        cmds.cmd_push_constants(0, &[1, 0, 0, 0], 256).unwrap();
        cmds.cmd_barrier(BarrierScope::Transfer, BarrierScope::ComputeShader).unwrap();
        cmds.cmd_dispatch(1, 1, 1).unwrap();
        cmds.cmd_barrier(BarrierScope::ComputeShader, BarrierScope::Host).unwrap();
        cmds.cmd_copy_buffer(src, dst, 0, 0, 4).unwrap();
        cmds.end().unwrap();

        // Submit and wait.
        let fence = engine.create_fence(false);
        engine.submit(cb, fence).unwrap();
        engine.wait_fence(fence, Duration::from_secs(1)).unwrap();

        // Verify transfer happened.
        let result = engine.buffer(dst).unwrap().read(0, 4).unwrap();
        assert_eq!(result, &[10, 20, 30, 40]);
        assert_eq!(engine.submission_count(), 1);
    }

    #[test]
    fn engine_submit_non_executable_fails() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let cb = engine.create_command_buffer();
        let fence = engine.create_fence(false);
        let err = engine.submit(cb, fence).unwrap_err();
        assert!(matches!(err, VulkanError::CommandBufferError(_)));
    }

    #[test]
    fn engine_invalid_command_buffer_submit() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let fence = engine.create_fence(false);
        let err = engine.submit(99, fence).unwrap_err();
        assert!(matches!(err, VulkanError::CommandBufferError(_)));
    }

    #[test]
    fn engine_invalid_fence_submit() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        let cb = engine.create_command_buffer();
        engine.command_buffer_mut(cb).unwrap().begin().unwrap();
        engine.command_buffer_mut(cb).unwrap().end().unwrap();
        let err = engine.submit(cb, 99).unwrap_err();
        assert!(matches!(err, VulkanError::CommandBufferError(_)));
    }

    #[test]
    fn engine_multiple_submissions() {
        let mut engine = VulkanComputeEngine::new(&default_config()).unwrap();
        for _ in 0..3 {
            let cb = engine.create_command_buffer();
            engine.command_buffer_mut(cb).unwrap().begin().unwrap();
            engine.command_buffer_mut(cb).unwrap().cmd_dispatch(1, 1, 1).unwrap();
            engine.command_buffer_mut(cb).unwrap().end().unwrap();
            let fence = engine.create_fence(false);
            engine.submit(cb, fence).unwrap();
        }
        assert_eq!(engine.submission_count(), 3);
    }

    #[test]
    fn engine_wait_unsignaled_fence_timeout() {
        let engine = VulkanComputeEngine::new(&default_config()).unwrap();
        // No fences created — but let's create one without signaling.
        let mut eng2 = VulkanComputeEngine::new(&default_config()).unwrap();
        let f = eng2.create_fence(false);
        let err = eng2.wait_fence(f, Duration::from_millis(1)).unwrap_err();
        assert_eq!(err, VulkanError::FenceTimeout);
        drop(engine);
    }

    #[test]
    fn engine_debug_messages_with_validation() {
        let mut engine = VulkanComputeEngine::new(&config_with_validation()).unwrap();
        let msgs = engine.instance_mut().drain_debug_messages();
        assert!(!msgs.is_empty());
    }

    // ── Error display tests ─────────────────────────────────────────────

    #[test]
    fn error_display_instance() {
        let e = VulkanError::InstanceCreationFailed("oops".to_string());
        assert!(e.to_string().contains("oops"));
    }

    #[test]
    fn error_display_no_device() {
        let e = VulkanError::NoSuitableDevice;
        assert!(e.to_string().contains("physical device"));
    }

    #[test]
    fn error_display_fence_timeout() {
        let e = VulkanError::FenceTimeout;
        assert!(e.to_string().contains("timed out"));
    }

    #[test]
    fn error_display_push_constant() {
        let e = VulkanError::PushConstantOverflow { actual: 300, limit: 256 };
        let s = e.to_string();
        assert!(s.contains("300"));
        assert!(s.contains("256"));
    }

    #[test]
    fn error_display_descriptor_binding() {
        let e = VulkanError::DescriptorBindingOutOfRange { binding: 5, max: 3 };
        let s = e.to_string();
        assert!(s.contains('5'));
        assert!(s.contains('3'));
    }

    #[test]
    fn error_display_device_lost() {
        let e = VulkanError::DeviceLost;
        assert!(e.to_string().contains("device lost"));
    }

    #[test]
    fn error_display_zero_work_groups() {
        let e = VulkanError::ZeroWorkGroups;
        assert!(e.to_string().contains("zero"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(VulkanError::DeviceLost);
        assert!(e.to_string().contains("device lost"));
    }

    // ── Hash helper tests ───────────────────────────────────────────────

    #[test]
    fn simple_hash_deterministic() {
        let data = b"hello world";
        assert_eq!(simple_hash(data), simple_hash(data));
    }

    #[test]
    fn simple_hash_different_inputs() {
        assert_ne!(simple_hash(b"aaa"), simple_hash(b"bbb"));
    }

    #[test]
    fn simple_hash_empty() {
        // Should not panic.
        let _ = simple_hash(b"");
    }

    // ── Proptest ────────────────────────────────────────────────────────

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn buffer_random_sizes(size in 1usize..=1_000_000) {
                let buf = VulkanBuffer::new(size, BufferMemoryType::HostVisible).unwrap();
                prop_assert_eq!(buf.size(), size);
            }

            #[test]
            fn dispatch_random_dimensions(
                x in 1u32..=256,
                y in 1u32..=256,
                z in 1u32..=64,
            ) {
                let mut cb = VulkanCommandBuffer::new();
                cb.begin().unwrap();
                cb.cmd_dispatch(x, y, z).unwrap();
                cb.end().unwrap();
                prop_assert_eq!(cb.command_count(), 1);
            }

            #[test]
            fn buffer_write_read_roundtrip(
                data in proptest::collection::vec(any::<u8>(), 1..=512),
            ) {
                let mut buf = VulkanBuffer::new(data.len(), BufferMemoryType::HostVisible).unwrap();
                buf.write(0, &data).unwrap();
                let read = buf.read(0, data.len()).unwrap();
                prop_assert_eq!(read, data.as_slice());
            }

            #[test]
            fn push_constants_within_limit(
                size in 1usize..=128,
            ) {
                let data = vec![0u8; size];
                let mut cb = VulkanCommandBuffer::new();
                cb.begin().unwrap();
                cb.cmd_push_constants(0, &data, 128).unwrap();
                cb.end().unwrap();
                prop_assert_eq!(cb.command_count(), 1);
            }

            #[test]
            fn fence_signal_then_wait(dummy in 0u8..1) {
                let _ = dummy;
                let fence = VulkanFence::new();
                fence.signal();
                fence.wait(Duration::from_millis(100)).unwrap();
                prop_assert!(fence.is_signaled());
            }
        }
    }
}
