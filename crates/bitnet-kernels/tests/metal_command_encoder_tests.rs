#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal command encoder infrastructure tests for Apple Silicon.
//!
//! Validates command buffer lifecycle, compute/blit/render encoder
//! configuration, parallel encoding, synchronization primitives,
//! indirect command buffers, resource usage tracking, status/error
//! handling, and GPU timeline semaphore simulation.
//!
//! All tests use mock/simulated Metal types — no GPU hardware required.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ── Metal constants ─────────────────────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD width on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Metal buffer alignment requirement (bytes).
const BUFFER_ALIGNMENT: usize = 256;

/// Maximum threadgroup memory per threadgroup (bytes).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

/// Maximum command buffers per queue (practical limit).
const MAX_COMMAND_BUFFERS_PER_QUEUE: usize = 64;

/// Maximum argument buffer entries.
const MAX_ARGUMENT_BUFFER_ENTRIES: usize = 31;

// ── Status / Error types ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommandBufferError {
    None,
    Internal(String),
    Timeout,
    PageFault,
    NotPermitted,
    OutOfMemory,
    InvalidResource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceUsage {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderType {
    Compute,
    Blit,
    Render,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DispatchType {
    Serial,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadAction {
    DontCare,
    Load,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreAction {
    DontCare,
    Store,
    MultisampleResolve,
}

// ── Mock Metal types ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct GpuBuffer {
    label: String,
    size: usize,
    data: Vec<u8>,
}

impl GpuBuffer {
    fn new(label: &str, size: usize) -> Self {
        Self { label: label.to_string(), size, data: vec![0u8; size] }
    }

    fn aligned_size(&self) -> usize {
        (self.size + BUFFER_ALIGNMENT - 1) & !(BUFFER_ALIGNMENT - 1)
    }
}

#[derive(Debug, Clone)]
struct Texture {
    label: String,
    width: u32,
    height: u32,
    pixel_format: u32,
}

#[derive(Debug)]
struct ComputePipelineState {
    label: String,
    max_total_threads: u32,
    threadgroup_memory_length: usize,
}

impl ComputePipelineState {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            max_total_threads: MAX_THREADS_PER_THREADGROUP,
            threadgroup_memory_length: 0,
        }
    }

    fn with_threadgroup_memory(mut self, bytes: usize) -> Self {
        self.threadgroup_memory_length = bytes;
        self
    }
}

#[derive(Debug)]
struct ResourceBinding {
    index: u32,
    usage: ResourceUsage,
    label: String,
}

#[derive(Debug)]
struct ComputeEncoder {
    label: String,
    is_ended: bool,
    dispatch_count: u32,
    pipeline: Option<String>,
    buffer_bindings: Vec<ResourceBinding>,
    threadgroup_memory_bytes: usize,
    dispatches: Vec<(u32, u32, u32)>,
}

impl ComputeEncoder {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            is_ended: false,
            dispatch_count: 0,
            pipeline: None,
            buffer_bindings: Vec::new(),
            threadgroup_memory_bytes: 0,
            dispatches: Vec::new(),
        }
    }

    fn set_pipeline(&mut self, pipeline: &ComputePipelineState) {
        assert!(!self.is_ended, "encoder already ended");
        self.pipeline = Some(pipeline.label.clone());
    }

    fn set_buffer(&mut self, index: u32, _buf: &GpuBuffer, usage: ResourceUsage) {
        assert!(!self.is_ended, "encoder already ended");
        self.buffer_bindings.push(ResourceBinding { index, usage, label: _buf.label.clone() });
    }

    fn set_threadgroup_memory(&mut self, bytes: usize, index: u32) {
        assert!(!self.is_ended, "encoder already ended");
        assert!(bytes <= MAX_THREADGROUP_MEMORY, "threadgroup memory exceeds limit");
        let _ = index;
        self.threadgroup_memory_bytes = bytes;
    }

    fn dispatch_threads(&mut self, grid: (u32, u32, u32), threadgroup: (u32, u32, u32)) {
        assert!(!self.is_ended, "encoder already ended");
        assert!(self.pipeline.is_some(), "no pipeline set");
        let total = threadgroup.0 * threadgroup.1 * threadgroup.2;
        assert!(total <= MAX_THREADS_PER_THREADGROUP, "threadgroup size exceeds limit");
        self.dispatches.push(grid);
        self.dispatch_count += 1;
    }

    fn dispatch_threadgroups(
        &mut self,
        groups: (u32, u32, u32),
        threads_per_group: (u32, u32, u32),
    ) {
        assert!(!self.is_ended, "encoder already ended");
        assert!(self.pipeline.is_some(), "no pipeline set");
        let total = threads_per_group.0 * threads_per_group.1 * threads_per_group.2;
        assert!(total <= MAX_THREADS_PER_THREADGROUP, "threadgroup size exceeds limit");
        let grid = (
            groups.0 * threads_per_group.0,
            groups.1 * threads_per_group.1,
            groups.2 * threads_per_group.2,
        );
        self.dispatches.push(grid);
        self.dispatch_count += 1;
    }

    fn end_encoding(&mut self) {
        assert!(!self.is_ended, "encoder already ended");
        self.is_ended = true;
    }
}

#[derive(Debug)]
struct BlitEncoder {
    label: String,
    is_ended: bool,
    copy_count: u32,
    fill_count: u32,
    operations: Vec<String>,
}

impl BlitEncoder {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            is_ended: false,
            copy_count: 0,
            fill_count: 0,
            operations: Vec::new(),
        }
    }

    fn copy_buffer(
        &mut self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &GpuBuffer,
        dst_offset: usize,
        size: usize,
    ) {
        assert!(!self.is_ended, "encoder already ended");
        assert!(src_offset + size <= src.size, "source overflow");
        assert!(dst_offset + size <= dst.size, "destination overflow");
        self.operations.push(format!(
            "copy {}[{}..{}] -> {}[{}..{}]",
            src.label,
            src_offset,
            src_offset + size,
            dst.label,
            dst_offset,
            dst_offset + size
        ));
        self.copy_count += 1;
    }

    fn fill_buffer(&mut self, buf: &GpuBuffer, range_offset: usize, range_size: usize, value: u8) {
        assert!(!self.is_ended, "encoder already ended");
        assert!(range_offset + range_size <= buf.size, "fill range overflow");
        self.operations.push(format!(
            "fill {}[{}..{}] = 0x{:02x}",
            buf.label,
            range_offset,
            range_offset + range_size,
            value
        ));
        self.fill_count += 1;
    }

    fn synchronize_resource(&mut self, buf: &GpuBuffer) {
        assert!(!self.is_ended, "encoder already ended");
        self.operations.push(format!("sync {}", buf.label));
    }

    fn end_encoding(&mut self) {
        assert!(!self.is_ended, "encoder already ended");
        self.is_ended = true;
    }
}

#[derive(Debug, Clone)]
struct ColorAttachment {
    texture: Option<Texture>,
    load_action: LoadAction,
    store_action: StoreAction,
    clear_color: (f64, f64, f64, f64),
}

impl Default for ColorAttachment {
    fn default() -> Self {
        Self {
            texture: None,
            load_action: LoadAction::DontCare,
            store_action: StoreAction::DontCare,
            clear_color: (0.0, 0.0, 0.0, 1.0),
        }
    }
}

#[derive(Debug)]
struct RenderPassDescriptor {
    label: String,
    color_attachments: Vec<ColorAttachment>,
    depth_attachment: Option<Texture>,
    stencil_attachment: Option<Texture>,
}

impl RenderPassDescriptor {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            color_attachments: Vec::new(),
            depth_attachment: None,
            stencil_attachment: None,
        }
    }

    fn add_color_attachment(&mut self, attachment: ColorAttachment) {
        self.color_attachments.push(attachment);
    }
}

#[derive(Debug)]
struct CommandBuffer {
    label: String,
    status: CommandBufferStatus,
    error: CommandBufferError,
    encoders_created: Vec<EncoderType>,
    active_encoder: Option<EncoderType>,
    enqueued_at: Option<u64>,
    committed_at: Option<u64>,
    completed_at: Option<u64>,
    retained_references: bool,
}

impl CommandBuffer {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            status: CommandBufferStatus::NotEnqueued,
            error: CommandBufferError::None,
            encoders_created: Vec::new(),
            active_encoder: None,
            enqueued_at: None,
            committed_at: None,
            completed_at: None,
            retained_references: true,
        }
    }

    fn make_compute_encoder(&mut self, label: &str) -> ComputeEncoder {
        assert!(self.active_encoder.is_none(), "another encoder is active");
        assert!(
            self.status == CommandBufferStatus::NotEnqueued
                || self.status == CommandBufferStatus::Enqueued,
            "cannot create encoder on committed buffer"
        );
        self.active_encoder = Some(EncoderType::Compute);
        self.encoders_created.push(EncoderType::Compute);
        ComputeEncoder::new(label)
    }

    fn make_blit_encoder(&mut self, label: &str) -> BlitEncoder {
        assert!(self.active_encoder.is_none(), "another encoder is active");
        self.active_encoder = Some(EncoderType::Blit);
        self.encoders_created.push(EncoderType::Blit);
        BlitEncoder::new(label)
    }

    fn end_encoder(&mut self, encoder_type: EncoderType) {
        assert_eq!(self.active_encoder, Some(encoder_type), "ending wrong encoder type");
        self.active_encoder = None;
    }

    fn enqueue(&mut self) {
        assert_eq!(self.status, CommandBufferStatus::NotEnqueued);
        self.status = CommandBufferStatus::Enqueued;
        self.enqueued_at = Some(1);
    }

    fn commit(&mut self, timestamp: u64) {
        assert!(
            self.status == CommandBufferStatus::NotEnqueued
                || self.status == CommandBufferStatus::Enqueued,
            "cannot commit buffer in state {:?}",
            self.status
        );
        assert!(self.active_encoder.is_none(), "cannot commit with active encoder");
        self.status = CommandBufferStatus::Committed;
        self.committed_at = Some(timestamp);
    }

    fn simulate_schedule(&mut self) {
        assert_eq!(self.status, CommandBufferStatus::Committed);
        self.status = CommandBufferStatus::Scheduled;
    }

    fn simulate_complete(&mut self, timestamp: u64) {
        assert!(
            self.status == CommandBufferStatus::Committed
                || self.status == CommandBufferStatus::Scheduled
        );
        self.status = CommandBufferStatus::Completed;
        self.completed_at = Some(timestamp);
    }

    fn simulate_error(&mut self, err: CommandBufferError) {
        self.status = CommandBufferStatus::Error;
        self.error = err;
    }
}

#[derive(Debug)]
struct CommandQueue {
    label: String,
    buffers: Vec<String>,
    max_buffers: usize,
}

impl CommandQueue {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            buffers: Vec::new(),
            max_buffers: MAX_COMMAND_BUFFERS_PER_QUEUE,
        }
    }

    fn make_command_buffer(&mut self, label: &str) -> CommandBuffer {
        assert!(self.buffers.len() < self.max_buffers, "command queue full");
        self.buffers.push(label.to_string());
        CommandBuffer::new(label)
    }
}

/// Simulates parallel command encoding via multiple command buffers.
struct ParallelEncoder {
    buffers: Vec<CommandBuffer>,
}

impl ParallelEncoder {
    fn new(count: usize) -> Self {
        let buffers = (0..count).map(|i| CommandBuffer::new(&format!("parallel-{i}"))).collect();
        Self { buffers }
    }

    fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    fn encode_all<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut CommandBuffer),
    {
        for (i, buf) in self.buffers.iter_mut().enumerate() {
            f(i, buf);
        }
    }

    fn commit_all(&mut self, base_time: u64) {
        for (i, buf) in self.buffers.iter_mut().enumerate() {
            if buf.active_encoder.is_some() {
                buf.active_encoder = None;
            }
            buf.commit(base_time + i as u64);
        }
    }
}

/// Indirect command buffer for GPU-driven dispatch.
#[derive(Debug)]
struct IndirectCommandBuffer {
    label: String,
    max_commands: usize,
    commands: Vec<IndirectCommand>,
    inherit_pipeline: bool,
    inherit_buffers: bool,
}

#[derive(Debug, Clone)]
struct IndirectCommand {
    pipeline_index: Option<u32>,
    kernel_buffer_index: u32,
    threadgroups: (u32, u32, u32),
    threads_per_threadgroup: (u32, u32, u32),
}

impl IndirectCommandBuffer {
    fn new(label: &str, max_commands: usize) -> Self {
        Self {
            label: label.to_string(),
            max_commands,
            commands: Vec::new(),
            inherit_pipeline: false,
            inherit_buffers: false,
        }
    }

    fn set_inherit_pipeline(&mut self, inherit: bool) {
        self.inherit_pipeline = inherit;
    }

    fn set_inherit_buffers(&mut self, inherit: bool) {
        self.inherit_buffers = inherit;
    }

    fn add_command(&mut self, cmd: IndirectCommand) -> bool {
        if self.commands.len() >= self.max_commands {
            return false;
        }
        self.commands.push(cmd);
        true
    }

    fn command_count(&self) -> usize {
        self.commands.len()
    }

    fn reset(&mut self) {
        self.commands.clear();
    }
}

/// Resource usage tracker for encoder validation.
#[derive(Debug)]
struct ResourceTracker {
    entries: Vec<(String, ResourceUsage, EncoderType)>,
    hazards: Vec<String>,
}

impl ResourceTracker {
    fn new() -> Self {
        Self { entries: Vec::new(), hazards: Vec::new() }
    }

    fn track(&mut self, resource: &str, usage: ResourceUsage, encoder: EncoderType) {
        // Detect write-after-write and read-after-write hazards.
        for (res, prev_usage, _) in &self.entries {
            if res == resource {
                match (prev_usage, &usage) {
                    (ResourceUsage::Write, ResourceUsage::Write)
                    | (ResourceUsage::Write, ResourceUsage::Read)
                    | (ResourceUsage::ReadWrite, ResourceUsage::Write) => {
                        self.hazards.push(format!(
                            "hazard on '{}': {:?} after {:?}",
                            resource, usage, prev_usage
                        ));
                    }
                    _ => {}
                }
            }
        }
        self.entries.push((resource.to_string(), usage, encoder));
    }

    fn has_hazards(&self) -> bool {
        !self.hazards.is_empty()
    }

    fn hazard_count(&self) -> usize {
        self.hazards.len()
    }
}

/// GPU timeline semaphore simulation for cross-queue synchronization.
struct TimelineSemaphore {
    value: Arc<AtomicU64>,
    label: String,
}

impl TimelineSemaphore {
    fn new(label: &str, initial: u64) -> Self {
        Self { value: Arc::new(AtomicU64::new(initial)), label: label.to_string() }
    }

    fn signal(&self, val: u64) {
        self.value.fetch_max(val, Ordering::Release);
    }

    fn current_value(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    fn wait_value(&self, target: u64) -> bool {
        // Simulated: just check if current >= target.
        self.current_value() >= target
    }
}

/// Fence for tracking command buffer completion order.
struct Fence {
    label: String,
    signaled: Arc<Mutex<bool>>,
}

impl Fence {
    fn new(label: &str) -> Self {
        Self { label: label.to_string(), signaled: Arc::new(Mutex::new(false)) }
    }

    fn signal(&self) {
        *self.signaled.lock().unwrap() = true;
    }

    fn is_signaled(&self) -> bool {
        *self.signaled.lock().unwrap()
    }

    fn reset(&self) {
        *self.signaled.lock().unwrap() = false;
    }
}

// ── Helper functions ────────────────────────────────────────────────

fn ceil_div(total: u32, group_size: u32) -> u32 {
    assert_ne!(group_size, 0);
    (total + group_size - 1) / group_size
}

fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

// =====================================================================
// 1. Command buffer creation and lifecycle
// =====================================================================

#[test]
fn test_command_buffer_initial_status() {
    let cb = CommandBuffer::new("test-buf");
    assert_eq!(cb.status, CommandBufferStatus::NotEnqueued);
    assert_eq!(cb.error, CommandBufferError::None);
    assert!(cb.active_encoder.is_none());
}

#[test]
fn test_command_buffer_label() {
    let cb = CommandBuffer::new("my-label");
    assert_eq!(cb.label, "my-label");
}

#[test]
fn test_command_buffer_enqueue() {
    let mut cb = CommandBuffer::new("enqueue-test");
    cb.enqueue();
    assert_eq!(cb.status, CommandBufferStatus::Enqueued);
    assert!(cb.enqueued_at.is_some());
}

#[test]
fn test_command_buffer_commit_from_not_enqueued() {
    let mut cb = CommandBuffer::new("commit-direct");
    cb.commit(100);
    assert_eq!(cb.status, CommandBufferStatus::Committed);
    assert_eq!(cb.committed_at, Some(100));
}

#[test]
fn test_command_buffer_commit_from_enqueued() {
    let mut cb = CommandBuffer::new("commit-enqueued");
    cb.enqueue();
    cb.commit(200);
    assert_eq!(cb.status, CommandBufferStatus::Committed);
}

#[test]
fn test_command_buffer_full_lifecycle() {
    let mut cb = CommandBuffer::new("lifecycle");
    assert_eq!(cb.status, CommandBufferStatus::NotEnqueued);
    cb.enqueue();
    assert_eq!(cb.status, CommandBufferStatus::Enqueued);
    cb.commit(10);
    assert_eq!(cb.status, CommandBufferStatus::Committed);
    cb.simulate_schedule();
    assert_eq!(cb.status, CommandBufferStatus::Scheduled);
    cb.simulate_complete(20);
    assert_eq!(cb.status, CommandBufferStatus::Completed);
    assert_eq!(cb.completed_at, Some(20));
}

#[test]
fn test_command_buffer_retained_references_default() {
    let cb = CommandBuffer::new("retained");
    assert!(cb.retained_references);
}

#[test]
fn test_command_buffer_no_retained_references() {
    let mut cb = CommandBuffer::new("no-retain");
    cb.retained_references = false;
    assert!(!cb.retained_references);
}

#[test]
fn test_command_queue_creates_buffers() {
    let mut q = CommandQueue::new("q0");
    let _cb1 = q.make_command_buffer("buf-1");
    let _cb2 = q.make_command_buffer("buf-2");
    assert_eq!(q.buffers.len(), 2);
}

#[test]
fn test_command_queue_label() {
    let q = CommandQueue::new("main-queue");
    assert_eq!(q.label, "main-queue");
}

// =====================================================================
// 2. Compute command encoder configuration
// =====================================================================

#[test]
fn test_compute_encoder_creation() {
    let mut cb = CommandBuffer::new("cb");
    let enc = cb.make_compute_encoder("compute-0");
    assert_eq!(enc.label, "compute-0");
    assert!(!enc.is_ended);
    assert_eq!(enc.dispatch_count, 0);
}

#[test]
fn test_compute_encoder_set_pipeline() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("matmul");
    enc.set_pipeline(&pso);
    assert_eq!(enc.pipeline.as_deref(), Some("matmul"));
}

#[test]
fn test_compute_encoder_buffer_binding() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let buf = GpuBuffer::new("input", 1024);
    enc.set_buffer(0, &buf, ResourceUsage::Read);
    assert_eq!(enc.buffer_bindings.len(), 1);
    assert_eq!(enc.buffer_bindings[0].index, 0);
}

#[test]
fn test_compute_encoder_multiple_bindings() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("kern");
    enc.set_pipeline(&pso);
    let a = GpuBuffer::new("a", 512);
    let b = GpuBuffer::new("b", 512);
    let c = GpuBuffer::new("c", 512);
    enc.set_buffer(0, &a, ResourceUsage::Read);
    enc.set_buffer(1, &b, ResourceUsage::Read);
    enc.set_buffer(2, &c, ResourceUsage::Write);
    assert_eq!(enc.buffer_bindings.len(), 3);
}

#[test]
fn test_compute_encoder_threadgroup_memory() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    enc.set_threadgroup_memory(4096, 0);
    assert_eq!(enc.threadgroup_memory_bytes, 4096);
}

#[test]
fn test_compute_encoder_dispatch_threads() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("kern");
    enc.set_pipeline(&pso);
    enc.dispatch_threads((256, 1, 1), (64, 1, 1));
    assert_eq!(enc.dispatch_count, 1);
    assert_eq!(enc.dispatches[0], (256, 1, 1));
}

#[test]
fn test_compute_encoder_dispatch_threadgroups() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("kern");
    enc.set_pipeline(&pso);
    enc.dispatch_threadgroups((4, 1, 1), (64, 1, 1));
    assert_eq!(enc.dispatch_count, 1);
    // 4 groups × 64 threads = 256 total
    assert_eq!(enc.dispatches[0], (256, 1, 1));
}

#[test]
fn test_compute_encoder_multiple_dispatches() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("kern");
    enc.set_pipeline(&pso);
    enc.dispatch_threads((128, 1, 1), (32, 1, 1));
    enc.dispatch_threads((256, 1, 1), (64, 1, 1));
    enc.dispatch_threads((512, 1, 1), (128, 1, 1));
    assert_eq!(enc.dispatch_count, 3);
}

#[test]
fn test_compute_encoder_end_encoding() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    enc.end_encoding();
    assert!(enc.is_ended);
    cb.end_encoder(EncoderType::Compute);
    assert!(cb.active_encoder.is_none());
}

#[test]
fn test_compute_encoder_2d_dispatch() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("img");
    enc.set_pipeline(&pso);
    enc.dispatch_threads((64, 64, 1), (8, 8, 1));
    assert_eq!(enc.dispatches[0], (64, 64, 1));
}

#[test]
fn test_compute_encoder_3d_dispatch() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("vol");
    enc.set_pipeline(&pso);
    enc.dispatch_threads((32, 32, 32), (4, 4, 4));
    assert_eq!(enc.dispatches[0], (32, 32, 32));
}

#[test]
fn test_compute_pipeline_state_defaults() {
    let pso = ComputePipelineState::new("default");
    assert_eq!(pso.max_total_threads, MAX_THREADS_PER_THREADGROUP);
    assert_eq!(pso.threadgroup_memory_length, 0);
}

#[test]
fn test_compute_pipeline_with_shared_memory() {
    let pso = ComputePipelineState::new("reduction").with_threadgroup_memory(8192);
    assert_eq!(pso.threadgroup_memory_length, 8192);
}

// =====================================================================
// 3. Blit command encoder for buffer copies
// =====================================================================

#[test]
fn test_blit_encoder_creation() {
    let mut cb = CommandBuffer::new("cb");
    let enc = cb.make_blit_encoder("blit-0");
    assert_eq!(enc.label, "blit-0");
    assert!(!enc.is_ended);
}

#[test]
fn test_blit_encoder_copy_buffer() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let src = GpuBuffer::new("src", 1024);
    let dst = GpuBuffer::new("dst", 1024);
    enc.copy_buffer(&src, 0, &dst, 0, 1024);
    assert_eq!(enc.copy_count, 1);
}

#[test]
fn test_blit_encoder_partial_copy() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let src = GpuBuffer::new("src", 2048);
    let dst = GpuBuffer::new("dst", 1024);
    enc.copy_buffer(&src, 512, &dst, 0, 512);
    assert_eq!(enc.copy_count, 1);
    assert!(enc.operations[0].contains("src[512..1024]"));
}

#[test]
fn test_blit_encoder_fill_buffer() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let buf = GpuBuffer::new("buf", 4096);
    enc.fill_buffer(&buf, 0, 4096, 0xFF);
    assert_eq!(enc.fill_count, 1);
}

#[test]
fn test_blit_encoder_synchronize() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let buf = GpuBuffer::new("shared", 256);
    enc.synchronize_resource(&buf);
    assert_eq!(enc.operations.len(), 1);
    assert!(enc.operations[0].contains("sync"));
}

#[test]
fn test_blit_encoder_multiple_copies() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let a = GpuBuffer::new("a", 1024);
    let b = GpuBuffer::new("b", 1024);
    let c = GpuBuffer::new("c", 1024);
    enc.copy_buffer(&a, 0, &b, 0, 512);
    enc.copy_buffer(&b, 0, &c, 0, 512);
    assert_eq!(enc.copy_count, 2);
}

#[test]
fn test_blit_encoder_copy_and_fill() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let a = GpuBuffer::new("a", 1024);
    let b = GpuBuffer::new("b", 1024);
    enc.fill_buffer(&a, 0, 1024, 0);
    enc.copy_buffer(&a, 0, &b, 0, 1024);
    assert_eq!(enc.fill_count, 1);
    assert_eq!(enc.copy_count, 1);
    assert_eq!(enc.operations.len(), 2);
}

#[test]
fn test_blit_encoder_end_encoding() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    enc.end_encoding();
    assert!(enc.is_ended);
    cb.end_encoder(EncoderType::Blit);
}

#[test]
fn test_blit_encoder_fill_partial_range() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let buf = GpuBuffer::new("buf", 4096);
    enc.fill_buffer(&buf, 256, 512, 0xAB);
    assert!(enc.operations[0].contains("buf[256..768]"));
}

// =====================================================================
// 4. Render pass descriptor setup
// =====================================================================

#[test]
fn test_render_pass_descriptor_creation() {
    let rpd = RenderPassDescriptor::new("pass-0");
    assert_eq!(rpd.label, "pass-0");
    assert!(rpd.color_attachments.is_empty());
    assert!(rpd.depth_attachment.is_none());
}

#[test]
fn test_render_pass_add_color_attachment() {
    let mut rpd = RenderPassDescriptor::new("pass");
    let tex = Texture {
        label: "color".into(),
        width: 1920,
        height: 1080,
        pixel_format: 80, // BGRA8Unorm
    };
    let attachment = ColorAttachment {
        texture: Some(tex),
        load_action: LoadAction::Clear,
        store_action: StoreAction::Store,
        clear_color: (0.0, 0.0, 0.0, 1.0),
    };
    rpd.add_color_attachment(attachment);
    assert_eq!(rpd.color_attachments.len(), 1);
}

#[test]
fn test_render_pass_multiple_color_attachments() {
    let mut rpd = RenderPassDescriptor::new("mrt");
    for i in 0..4 {
        let tex =
            Texture { label: format!("color-{i}"), width: 512, height: 512, pixel_format: 80 };
        rpd.add_color_attachment(ColorAttachment {
            texture: Some(tex),
            load_action: LoadAction::Clear,
            store_action: StoreAction::Store,
            clear_color: (0.0, 0.0, 0.0, 1.0),
        });
    }
    assert_eq!(rpd.color_attachments.len(), 4);
}

#[test]
fn test_render_pass_depth_attachment() {
    let mut rpd = RenderPassDescriptor::new("depth");
    rpd.depth_attachment = Some(Texture {
        label: "depth-tex".into(),
        width: 1920,
        height: 1080,
        pixel_format: 252, // Depth32Float
    });
    assert!(rpd.depth_attachment.is_some());
}

#[test]
fn test_render_pass_stencil_attachment() {
    let mut rpd = RenderPassDescriptor::new("stencil");
    rpd.stencil_attachment = Some(Texture {
        label: "stencil-tex".into(),
        width: 512,
        height: 512,
        pixel_format: 253, // Stencil8
    });
    assert!(rpd.stencil_attachment.is_some());
}

#[test]
fn test_render_pass_clear_color_values() {
    let attachment = ColorAttachment {
        texture: None,
        load_action: LoadAction::Clear,
        store_action: StoreAction::Store,
        clear_color: (0.2, 0.4, 0.6, 1.0),
    };
    assert_eq!(attachment.clear_color, (0.2, 0.4, 0.6, 1.0));
}

#[test]
fn test_render_pass_load_action_dont_care() {
    let attachment = ColorAttachment { load_action: LoadAction::DontCare, ..Default::default() };
    assert_eq!(attachment.load_action, LoadAction::DontCare);
}

#[test]
fn test_render_pass_store_action_multisample_resolve() {
    let attachment =
        ColorAttachment { store_action: StoreAction::MultisampleResolve, ..Default::default() };
    assert_eq!(attachment.store_action, StoreAction::MultisampleResolve);
}

// =====================================================================
// 5. Parallel command encoding
// =====================================================================

#[test]
fn test_parallel_encoder_creation() {
    let pe = ParallelEncoder::new(4);
    assert_eq!(pe.buffer_count(), 4);
}

#[test]
fn test_parallel_encoder_encode_all() {
    let mut pe = ParallelEncoder::new(3);
    pe.encode_all(|i, buf| {
        let mut enc = buf.make_compute_encoder(&format!("enc-{i}"));
        let pso = ComputePipelineState::new("kern");
        enc.set_pipeline(&pso);
        enc.dispatch_threads((64, 1, 1), (32, 1, 1));
        enc.end_encoding();
        buf.end_encoder(EncoderType::Compute);
    });
    for buf in &pe.buffers {
        assert_eq!(buf.encoders_created.len(), 1);
    }
}

#[test]
fn test_parallel_encoder_commit_all() {
    let mut pe = ParallelEncoder::new(3);
    pe.commit_all(100);
    for buf in &pe.buffers {
        assert_eq!(buf.status, CommandBufferStatus::Committed);
    }
}

#[test]
fn test_parallel_encoder_commit_timestamps() {
    let mut pe = ParallelEncoder::new(4);
    pe.commit_all(50);
    assert_eq!(pe.buffers[0].committed_at, Some(50));
    assert_eq!(pe.buffers[1].committed_at, Some(51));
    assert_eq!(pe.buffers[2].committed_at, Some(52));
    assert_eq!(pe.buffers[3].committed_at, Some(53));
}

#[test]
fn test_parallel_encoder_independent_buffers() {
    let mut pe = ParallelEncoder::new(2);
    pe.encode_all(|i, buf| {
        if i == 0 {
            let mut enc = buf.make_compute_encoder("compute");
            enc.end_encoding();
            buf.end_encoder(EncoderType::Compute);
        } else {
            let mut enc = buf.make_blit_encoder("blit");
            enc.end_encoding();
            buf.end_encoder(EncoderType::Blit);
        }
    });
    assert_eq!(pe.buffers[0].encoders_created[0], EncoderType::Compute);
    assert_eq!(pe.buffers[1].encoders_created[0], EncoderType::Blit);
}

#[test]
fn test_parallel_encoder_large_batch() {
    let mut pe = ParallelEncoder::new(16);
    pe.encode_all(|_i, buf| {
        let mut enc = buf.make_compute_encoder("enc");
        let pso = ComputePipelineState::new("kern");
        enc.set_pipeline(&pso);
        enc.dispatch_threads((1024, 1, 1), (256, 1, 1));
        enc.end_encoding();
        buf.end_encoder(EncoderType::Compute);
    });
    pe.commit_all(0);
    assert!(pe.buffers.iter().all(|b| b.status == CommandBufferStatus::Committed));
}

// =====================================================================
// 6. Command buffer synchronization
// =====================================================================

#[test]
fn test_fence_creation() {
    let fence = Fence::new("f0");
    assert_eq!(fence.label, "f0");
    assert!(!fence.is_signaled());
}

#[test]
fn test_fence_signal() {
    let fence = Fence::new("f");
    fence.signal();
    assert!(fence.is_signaled());
}

#[test]
fn test_fence_reset() {
    let fence = Fence::new("f");
    fence.signal();
    fence.reset();
    assert!(!fence.is_signaled());
}

#[test]
fn test_synchronization_ordered_completion() {
    let mut cb1 = CommandBuffer::new("first");
    let mut cb2 = CommandBuffer::new("second");
    cb1.commit(10);
    cb1.simulate_schedule();
    cb1.simulate_complete(20);
    cb2.commit(15);
    cb2.simulate_schedule();
    cb2.simulate_complete(25);
    assert!(cb1.completed_at.unwrap() < cb2.completed_at.unwrap());
}

#[test]
fn test_synchronization_wait_for_fence() {
    let fence = Fence::new("sync");
    // Simulate: cb1 signals fence on completion.
    let mut cb1 = CommandBuffer::new("producer");
    cb1.commit(10);
    cb1.simulate_complete(20);
    fence.signal();
    // Consumer waits.
    assert!(fence.is_signaled());
    let mut cb2 = CommandBuffer::new("consumer");
    cb2.commit(25);
    cb2.simulate_complete(30);
    assert_eq!(cb2.status, CommandBufferStatus::Completed);
}

#[test]
fn test_synchronization_multiple_fences() {
    let fences: Vec<Fence> = (0..4).map(|i| Fence::new(&format!("f-{i}"))).collect();
    for f in &fences {
        assert!(!f.is_signaled());
    }
    fences[0].signal();
    fences[2].signal();
    assert!(fences[0].is_signaled());
    assert!(!fences[1].is_signaled());
    assert!(fences[2].is_signaled());
    assert!(!fences[3].is_signaled());
}

#[test]
fn test_synchronization_command_buffer_ordering() {
    let mut buffers: Vec<CommandBuffer> =
        (0..5).map(|i| CommandBuffer::new(&format!("cb-{i}"))).collect();
    for (i, buf) in buffers.iter_mut().enumerate() {
        buf.commit(i as u64 * 10);
    }
    for (i, buf) in buffers.iter_mut().enumerate() {
        buf.simulate_complete(i as u64 * 10 + 5);
    }
    // Verify monotonic completion.
    for i in 1..buffers.len() {
        assert!(buffers[i].completed_at.unwrap() > buffers[i - 1].completed_at.unwrap());
    }
}

// =====================================================================
// 7. Indirect command buffers
// =====================================================================

#[test]
fn test_indirect_command_buffer_creation() {
    let icb = IndirectCommandBuffer::new("icb-0", 128);
    assert_eq!(icb.label, "icb-0");
    assert_eq!(icb.max_commands, 128);
    assert_eq!(icb.command_count(), 0);
}

#[test]
fn test_indirect_command_buffer_add_commands() {
    let mut icb = IndirectCommandBuffer::new("icb", 10);
    let cmd = IndirectCommand {
        pipeline_index: Some(0),
        kernel_buffer_index: 0,
        threadgroups: (4, 1, 1),
        threads_per_threadgroup: (64, 1, 1),
    };
    assert!(icb.add_command(cmd));
    assert_eq!(icb.command_count(), 1);
}

#[test]
fn test_indirect_command_buffer_capacity_limit() {
    let mut icb = IndirectCommandBuffer::new("icb", 2);
    let cmd = IndirectCommand {
        pipeline_index: None,
        kernel_buffer_index: 0,
        threadgroups: (1, 1, 1),
        threads_per_threadgroup: (32, 1, 1),
    };
    assert!(icb.add_command(cmd.clone()));
    assert!(icb.add_command(cmd.clone()));
    assert!(!icb.add_command(cmd));
}

#[test]
fn test_indirect_command_buffer_reset() {
    let mut icb = IndirectCommandBuffer::new("icb", 10);
    let cmd = IndirectCommand {
        pipeline_index: Some(0),
        kernel_buffer_index: 0,
        threadgroups: (2, 1, 1),
        threads_per_threadgroup: (64, 1, 1),
    };
    icb.add_command(cmd);
    assert_eq!(icb.command_count(), 1);
    icb.reset();
    assert_eq!(icb.command_count(), 0);
}

#[test]
fn test_indirect_command_buffer_inherit_pipeline() {
    let mut icb = IndirectCommandBuffer::new("icb", 32);
    icb.set_inherit_pipeline(true);
    assert!(icb.inherit_pipeline);
}

#[test]
fn test_indirect_command_buffer_inherit_buffers() {
    let mut icb = IndirectCommandBuffer::new("icb", 32);
    icb.set_inherit_buffers(true);
    assert!(icb.inherit_buffers);
}

#[test]
fn test_indirect_command_varying_threadgroups() {
    let mut icb = IndirectCommandBuffer::new("icb", 100);
    for i in 1..=5 {
        let cmd = IndirectCommand {
            pipeline_index: Some(0),
            kernel_buffer_index: 0,
            threadgroups: (i, 1, 1),
            threads_per_threadgroup: (32, 1, 1),
        };
        icb.add_command(cmd);
    }
    assert_eq!(icb.command_count(), 5);
}

#[test]
fn test_indirect_command_no_pipeline_index() {
    let mut icb = IndirectCommandBuffer::new("icb", 10);
    icb.set_inherit_pipeline(true);
    let cmd = IndirectCommand {
        pipeline_index: None,
        kernel_buffer_index: 0,
        threadgroups: (1, 1, 1),
        threads_per_threadgroup: (64, 1, 1),
    };
    assert!(icb.add_command(cmd));
    assert!(icb.commands[0].pipeline_index.is_none());
}

// =====================================================================
// 8. Encoder resource usage tracking
// =====================================================================

#[test]
fn test_resource_tracker_creation() {
    let rt = ResourceTracker::new();
    assert!(!rt.has_hazards());
    assert_eq!(rt.hazard_count(), 0);
}

#[test]
fn test_resource_tracker_single_read() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::Read, EncoderType::Compute);
    assert!(!rt.has_hazards());
    assert_eq!(rt.entries.len(), 1);
}

#[test]
fn test_resource_tracker_multiple_reads_no_hazard() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::Read, EncoderType::Compute);
    rt.track("buf-a", ResourceUsage::Read, EncoderType::Compute);
    assert!(!rt.has_hazards());
}

#[test]
fn test_resource_tracker_write_after_write_hazard() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::Write, EncoderType::Compute);
    rt.track("buf-a", ResourceUsage::Write, EncoderType::Blit);
    assert!(rt.has_hazards());
    assert_eq!(rt.hazard_count(), 1);
}

#[test]
fn test_resource_tracker_read_after_write_hazard() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::Write, EncoderType::Compute);
    rt.track("buf-a", ResourceUsage::Read, EncoderType::Compute);
    assert!(rt.has_hazards());
}

#[test]
fn test_resource_tracker_different_resources_no_hazard() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::Write, EncoderType::Compute);
    rt.track("buf-b", ResourceUsage::Write, EncoderType::Compute);
    assert!(!rt.has_hazards());
}

#[test]
fn test_resource_tracker_readwrite_then_write_hazard() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::ReadWrite, EncoderType::Compute);
    rt.track("buf-a", ResourceUsage::Write, EncoderType::Blit);
    assert!(rt.has_hazards());
}

#[test]
fn test_resource_tracker_mixed_resources() {
    let mut rt = ResourceTracker::new();
    rt.track("buf-a", ResourceUsage::Read, EncoderType::Compute);
    rt.track("buf-b", ResourceUsage::Write, EncoderType::Compute);
    rt.track("buf-c", ResourceUsage::Read, EncoderType::Blit);
    assert!(!rt.has_hazards());
    assert_eq!(rt.entries.len(), 3);
}

#[test]
fn test_resource_tracker_multi_encoder_types() {
    let mut rt = ResourceTracker::new();
    rt.track("weights", ResourceUsage::Read, EncoderType::Compute);
    rt.track("output", ResourceUsage::Write, EncoderType::Compute);
    rt.track("output", ResourceUsage::Read, EncoderType::Blit);
    // Write then read on "output" is a hazard.
    assert!(rt.has_hazards());
}

// =====================================================================
// 9. Command buffer status and error handling
// =====================================================================

#[test]
fn test_error_none_by_default() {
    let cb = CommandBuffer::new("cb");
    assert_eq!(cb.error, CommandBufferError::None);
}

#[test]
fn test_error_internal() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_error(CommandBufferError::Internal("GPU hang".to_string()));
    assert_eq!(cb.status, CommandBufferStatus::Error);
    assert_eq!(cb.error, CommandBufferError::Internal("GPU hang".to_string()));
}

#[test]
fn test_error_timeout() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_error(CommandBufferError::Timeout);
    assert_eq!(cb.error, CommandBufferError::Timeout);
}

#[test]
fn test_error_page_fault() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_error(CommandBufferError::PageFault);
    assert_eq!(cb.error, CommandBufferError::PageFault);
}

#[test]
fn test_error_not_permitted() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_error(CommandBufferError::NotPermitted);
    assert_eq!(cb.error, CommandBufferError::NotPermitted);
}

#[test]
fn test_error_out_of_memory() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_error(CommandBufferError::OutOfMemory);
    assert_eq!(cb.error, CommandBufferError::OutOfMemory);
}

#[test]
fn test_error_invalid_resource() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_error(CommandBufferError::InvalidResource);
    assert_eq!(cb.error, CommandBufferError::InvalidResource);
}

#[test]
fn test_status_after_complete() {
    let mut cb = CommandBuffer::new("cb");
    cb.commit(0);
    cb.simulate_complete(10);
    assert_eq!(cb.status, CommandBufferStatus::Completed);
    assert_eq!(cb.error, CommandBufferError::None);
}

#[test]
fn test_status_transitions_not_enqueued_to_committed() {
    let mut cb = CommandBuffer::new("cb");
    assert_eq!(cb.status, CommandBufferStatus::NotEnqueued);
    cb.commit(0);
    assert_eq!(cb.status, CommandBufferStatus::Committed);
}

#[test]
fn test_command_buffer_timestamps() {
    let mut cb = CommandBuffer::new("cb");
    assert!(cb.enqueued_at.is_none());
    assert!(cb.committed_at.is_none());
    assert!(cb.completed_at.is_none());
    cb.enqueue();
    assert!(cb.enqueued_at.is_some());
    cb.commit(50);
    assert_eq!(cb.committed_at, Some(50));
    cb.simulate_complete(100);
    assert_eq!(cb.completed_at, Some(100));
}

// =====================================================================
// 10. GPU timeline semaphore simulation
// =====================================================================

#[test]
fn test_timeline_semaphore_creation() {
    let sem = TimelineSemaphore::new("sem-0", 0);
    assert_eq!(sem.label, "sem-0");
    assert_eq!(sem.current_value(), 0);
}

#[test]
fn test_timeline_semaphore_signal() {
    let sem = TimelineSemaphore::new("sem", 0);
    sem.signal(5);
    assert_eq!(sem.current_value(), 5);
}

#[test]
fn test_timeline_semaphore_monotonic() {
    let sem = TimelineSemaphore::new("sem", 0);
    sem.signal(3);
    sem.signal(1); // Lower value ignored by fetch_max.
    assert_eq!(sem.current_value(), 3);
}

#[test]
fn test_timeline_semaphore_wait_ready() {
    let sem = TimelineSemaphore::new("sem", 10);
    assert!(sem.wait_value(5));
    assert!(sem.wait_value(10));
}

#[test]
fn test_timeline_semaphore_wait_not_ready() {
    let sem = TimelineSemaphore::new("sem", 5);
    assert!(!sem.wait_value(10));
}

#[test]
fn test_timeline_semaphore_sequential_signals() {
    let sem = TimelineSemaphore::new("sem", 0);
    for i in 1..=10 {
        sem.signal(i);
        assert_eq!(sem.current_value(), i);
    }
}

#[test]
fn test_timeline_semaphore_cross_queue_sync() {
    let sem = TimelineSemaphore::new("cross-queue", 0);
    // Queue A: produce work, then signal.
    let mut cb_a = CommandBuffer::new("producer");
    cb_a.commit(0);
    cb_a.simulate_complete(10);
    sem.signal(1);
    // Queue B: wait for semaphore, then consume.
    assert!(sem.wait_value(1));
    let mut cb_b = CommandBuffer::new("consumer");
    cb_b.commit(11);
    cb_b.simulate_complete(20);
    assert_eq!(cb_b.status, CommandBufferStatus::Completed);
}

#[test]
fn test_timeline_semaphore_multiple_producers() {
    let sem = TimelineSemaphore::new("multi", 0);
    sem.signal(2);
    sem.signal(5);
    sem.signal(3);
    // Should be max(2, 5, 3) = 5.
    assert_eq!(sem.current_value(), 5);
}

#[test]
fn test_timeline_semaphore_shared_ref() {
    let sem = TimelineSemaphore::new("shared", 0);
    let val = Arc::clone(&sem.value);
    sem.signal(42);
    assert_eq!(val.load(Ordering::Acquire), 42);
}

// =====================================================================
// Additional integration / edge-case tests
// =====================================================================

#[test]
fn test_sequential_encoder_types_on_single_buffer() {
    let mut cb = CommandBuffer::new("seq");
    // Compute encoder first.
    let mut ce = cb.make_compute_encoder("c0");
    let pso = ComputePipelineState::new("k");
    ce.set_pipeline(&pso);
    ce.dispatch_threads((64, 1, 1), (32, 1, 1));
    ce.end_encoding();
    cb.end_encoder(EncoderType::Compute);
    // Then blit encoder.
    let mut be = cb.make_blit_encoder("b0");
    let buf = GpuBuffer::new("x", 256);
    be.fill_buffer(&buf, 0, 256, 0);
    be.end_encoding();
    cb.end_encoder(EncoderType::Blit);
    assert_eq!(cb.encoders_created.len(), 2);
}

#[test]
fn test_gpu_buffer_aligned_size() {
    let buf = GpuBuffer::new("b", 300);
    assert_eq!(buf.size, 300);
    assert_eq!(buf.aligned_size(), 512); // next 256 boundary
}

#[test]
fn test_gpu_buffer_exact_alignment() {
    let buf = GpuBuffer::new("b", 256);
    assert_eq!(buf.aligned_size(), 256);
}

#[test]
fn test_gpu_buffer_zero_size() {
    let buf = GpuBuffer::new("empty", 0);
    assert_eq!(buf.aligned_size(), 0);
}

#[test]
fn test_ceil_div_basic() {
    assert_eq!(ceil_div(256, 64), 4);
    assert_eq!(ceil_div(257, 64), 5);
    assert_eq!(ceil_div(1, 32), 1);
}

#[test]
fn test_align_up_basic() {
    assert_eq!(align_up(100, 256), 256);
    assert_eq!(align_up(256, 256), 256);
    assert_eq!(align_up(257, 256), 512);
    assert_eq!(align_up(0, 256), 0);
}

#[test]
#[should_panic(expected = "threadgroup size exceeds limit")]
fn test_compute_dispatch_exceeds_threadgroup_limit() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("kern");
    enc.set_pipeline(&pso);
    // 2048 > MAX_THREADS_PER_THREADGROUP (1024)
    enc.dispatch_threads((2048, 1, 1), (2048, 1, 1));
}

#[test]
#[should_panic(expected = "threadgroup memory exceeds limit")]
fn test_compute_encoder_excessive_threadgroup_memory() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    enc.set_threadgroup_memory(MAX_THREADGROUP_MEMORY + 1, 0);
}

#[test]
#[should_panic(expected = "encoder already ended")]
fn test_compute_dispatch_after_end() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    let pso = ComputePipelineState::new("k");
    enc.set_pipeline(&pso);
    enc.end_encoding();
    enc.dispatch_threads((32, 1, 1), (32, 1, 1));
}

#[test]
#[should_panic(expected = "encoder already ended")]
fn test_blit_copy_after_end() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    enc.end_encoding();
    let a = GpuBuffer::new("a", 64);
    let b = GpuBuffer::new("b", 64);
    enc.copy_buffer(&a, 0, &b, 0, 64);
}

#[test]
#[should_panic(expected = "source overflow")]
fn test_blit_copy_source_overflow() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let src = GpuBuffer::new("src", 64);
    let dst = GpuBuffer::new("dst", 256);
    enc.copy_buffer(&src, 0, &dst, 0, 128); // src only 64 bytes
}

#[test]
#[should_panic(expected = "destination overflow")]
fn test_blit_copy_destination_overflow() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_blit_encoder("blit");
    let src = GpuBuffer::new("src", 256);
    let dst = GpuBuffer::new("dst", 64);
    enc.copy_buffer(&src, 0, &dst, 0, 128); // dst only 64 bytes
}

#[test]
#[should_panic(expected = "no pipeline set")]
fn test_compute_dispatch_without_pipeline() {
    let mut cb = CommandBuffer::new("cb");
    let mut enc = cb.make_compute_encoder("enc");
    enc.dispatch_threads((32, 1, 1), (32, 1, 1));
}

#[test]
#[should_panic(expected = "another encoder is active")]
fn test_two_encoders_simultaneously() {
    let mut cb = CommandBuffer::new("cb");
    let _enc1 = cb.make_compute_encoder("first");
    let _enc2 = cb.make_blit_encoder("second"); // Should panic
}

#[test]
#[should_panic(expected = "cannot commit with active encoder")]
fn test_commit_with_active_encoder() {
    let mut cb = CommandBuffer::new("cb");
    let _enc = cb.make_compute_encoder("enc");
    cb.commit(0); // Should panic — encoder not ended
}

#[test]
fn test_command_queue_max_buffers() {
    let mut q = CommandQueue::new("q");
    for i in 0..MAX_COMMAND_BUFFERS_PER_QUEUE {
        let _cb = q.make_command_buffer(&format!("cb-{i}"));
    }
    assert_eq!(q.buffers.len(), MAX_COMMAND_BUFFERS_PER_QUEUE);
}

#[test]
#[should_panic(expected = "command queue full")]
fn test_command_queue_overflow() {
    let mut q = CommandQueue::new("q");
    q.max_buffers = 2;
    let _cb1 = q.make_command_buffer("a");
    let _cb2 = q.make_command_buffer("b");
    let _cb3 = q.make_command_buffer("c"); // Should panic
}

#[test]
fn test_end_to_end_compute_workflow() {
    let mut q = CommandQueue::new("main");
    let mut cb = q.make_command_buffer("workflow");

    // Encode compute pass.
    let mut enc = cb.make_compute_encoder("matmul");
    let pso = ComputePipelineState::new("matmul_f32");
    enc.set_pipeline(&pso);
    let input = GpuBuffer::new("input", 4096);
    let output = GpuBuffer::new("output", 4096);
    enc.set_buffer(0, &input, ResourceUsage::Read);
    enc.set_buffer(1, &output, ResourceUsage::Write);
    let groups = ceil_div(1024, 256);
    enc.dispatch_threadgroups((groups, 1, 1), (256, 1, 1));
    enc.end_encoding();
    cb.end_encoder(EncoderType::Compute);

    // Encode blit pass to read back.
    let mut blit = cb.make_blit_encoder("readback");
    let staging = GpuBuffer::new("staging", 4096);
    blit.copy_buffer(&output, 0, &staging, 0, 4096);
    blit.end_encoding();
    cb.end_encoder(EncoderType::Blit);

    // Commit and simulate completion.
    cb.commit(0);
    cb.simulate_schedule();
    cb.simulate_complete(100);

    assert_eq!(cb.status, CommandBufferStatus::Completed);
    assert_eq!(cb.encoders_created.len(), 2);
}

#[test]
fn test_multiple_dispatch_with_resource_tracking() {
    let mut tracker = ResourceTracker::new();

    // Pass 1: write to output.
    tracker.track("output", ResourceUsage::Write, EncoderType::Compute);
    // Pass 2: read output as input to second kernel.
    tracker.track("output", ResourceUsage::Read, EncoderType::Compute);

    // Write-then-read is a hazard without explicit barrier.
    assert!(tracker.has_hazards());
    assert_eq!(tracker.hazard_count(), 1);
}

#[test]
fn test_fence_based_synchronization_pattern() {
    let fence = Fence::new("compute-done");

    let mut cb1 = CommandBuffer::new("compute-pass");
    cb1.commit(0);
    cb1.simulate_complete(10);
    fence.signal();

    assert!(fence.is_signaled());

    let mut cb2 = CommandBuffer::new("blit-pass");
    cb2.commit(11);
    cb2.simulate_complete(15);
    assert_eq!(cb2.status, CommandBufferStatus::Completed);
}

#[test]
fn test_indirect_command_buffer_full_workflow() {
    let mut icb = IndirectCommandBuffer::new("icb", 64);
    icb.set_inherit_pipeline(true);
    icb.set_inherit_buffers(true);

    for i in 0..32 {
        let groups = (i % 8) + 1;
        icb.add_command(IndirectCommand {
            pipeline_index: None,
            kernel_buffer_index: 0,
            threadgroups: (groups, 1, 1),
            threads_per_threadgroup: (64, 1, 1),
        });
    }
    assert_eq!(icb.command_count(), 32);
    assert!(icb.inherit_pipeline);
    assert!(icb.inherit_buffers);
}

#[test]
fn test_timeline_semaphore_pipeline_stages() {
    let vertex_done = TimelineSemaphore::new("vertex", 0);
    let fragment_done = TimelineSemaphore::new("fragment", 0);

    // Simulate pipeline: vertex → fragment → present.
    vertex_done.signal(1);
    assert!(vertex_done.wait_value(1));

    fragment_done.signal(1);
    assert!(fragment_done.wait_value(1));

    // Both stages complete.
    assert!(vertex_done.wait_value(1) && fragment_done.wait_value(1));
}

#[test]
fn test_buffer_data_initialized_to_zero() {
    let buf = GpuBuffer::new("zeroed", 128);
    assert!(buf.data.iter().all(|&b| b == 0));
}

#[test]
fn test_texture_creation() {
    let tex =
        Texture { label: "render-target".into(), width: 1920, height: 1080, pixel_format: 80 };
    assert_eq!(tex.width, 1920);
    assert_eq!(tex.height, 1080);
}
