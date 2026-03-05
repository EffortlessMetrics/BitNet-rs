#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal command encoder validation tests for Apple Silicon.
//!
//! Tests compute command encoder lifecycle, buffer binding, threadgroup
//! configuration, and error handling without requiring actual Metal hardware.

#![cfg(target_os = "macos")]
#![allow(clippy::float_cmp, clippy::needless_range_loop, clippy::too_many_arguments)]

// ============================================================================
// Mock Metal types
// ============================================================================

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Metal GPU resource usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ResourceUsage {
    Read,
    Write,
    ReadWrite,
}

/// Metal texture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureType {
    Texture1D,
    Texture2D,
    Texture3D,
}

/// Metal pixel format subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PixelFormat {
    R32Float,
    RGBA8Unorm,
    RGBA16Float,
    R16Float,
}

/// Metal sampler address mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SamplerAddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Metal sampler min/mag filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SamplerFilter {
    Nearest,
    Linear,
}

/// Errors from encoder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EncoderError {
    NotEncoding,
    AlreadyEncoding,
    AlreadyEnded,
    BufferSlotOutOfRange { slot: usize, max: usize },
    BufferOffsetMisaligned { offset: usize, alignment: usize },
    BufferOffsetOutOfBounds { offset: usize, length: usize },
    ThreadgroupExceedsMax { requested: usize, max: usize },
    ThreadgroupDimensionZero,
    ThreadgroupMemoryExceedsMax { requested: usize, max: usize },
    NoPipelineSet,
    TextureSlotOutOfRange { slot: usize, max: usize },
    SamplerSlotOutOfRange { slot: usize, max: usize },
    ResourceHazard { resource_id: u64 },
    EncoderReuseForbidden,
    GridSizeZero,
    DispatchWithoutPipeline,
    InvalidThreadsPerThreadgroup,
    NestedEncodingForbidden,
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEncoding => write!(f, "encoder is not in encoding state"),
            Self::AlreadyEncoding => write!(f, "encoder is already in encoding state"),
            Self::AlreadyEnded => write!(f, "encoder has already ended"),
            Self::BufferSlotOutOfRange { slot, max } => {
                write!(f, "buffer slot {slot} exceeds max {max}")
            }
            Self::BufferOffsetMisaligned { offset, alignment } => {
                write!(f, "buffer offset {offset} not aligned to {alignment}")
            }
            Self::BufferOffsetOutOfBounds { offset, length } => {
                write!(f, "buffer offset {offset} out of bounds (length {length})")
            }
            Self::ThreadgroupExceedsMax { requested, max } => {
                write!(f, "threadgroup size {requested} exceeds max {max}")
            }
            Self::ThreadgroupDimensionZero => write!(f, "threadgroup dimension must be non-zero"),
            Self::ThreadgroupMemoryExceedsMax { requested, max } => {
                write!(f, "threadgroup memory {requested} bytes exceeds max {max} bytes")
            }
            Self::NoPipelineSet => write!(f, "no compute pipeline state set"),
            Self::TextureSlotOutOfRange { slot, max } => {
                write!(f, "texture slot {slot} exceeds max {max}")
            }
            Self::SamplerSlotOutOfRange { slot, max } => {
                write!(f, "sampler slot {slot} exceeds max {max}")
            }
            Self::ResourceHazard { resource_id } => {
                write!(f, "resource hazard on resource {resource_id}")
            }
            Self::EncoderReuseForbidden => {
                write!(f, "encoder cannot be reused after ending")
            }
            Self::GridSizeZero => write!(f, "grid size must be non-zero in all dimensions"),
            Self::DispatchWithoutPipeline => {
                write!(f, "dispatch called without a pipeline state set")
            }
            Self::InvalidThreadsPerThreadgroup => {
                write!(f, "threads per threadgroup invalid")
            }
            Self::NestedEncodingForbidden => {
                write!(f, "nested encoding passes are forbidden")
            }
        }
    }
}

/// Mock Metal buffer.
#[derive(Debug, Clone)]
struct MockBuffer {
    id: u64,
    length: usize,
    label: Option<String>,
}

impl MockBuffer {
    fn new(length: usize) -> Self {
        Self { id: next_id(), length, label: None }
    }

    fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

/// Mock Metal texture.
#[derive(Debug, Clone)]
struct MockTexture {
    id: u64,
    texture_type: TextureType,
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    depth: u32,
}

impl MockTexture {
    fn new(texture_type: TextureType, pixel_format: PixelFormat, width: u32) -> Self {
        Self { id: next_id(), texture_type, pixel_format, width, height: 1, depth: 1 }
    }

    fn with_size(mut self, width: u32, height: u32, depth: u32) -> Self {
        self.width = width;
        self.height = height;
        self.depth = depth;
        self
    }
}

/// Mock Metal sampler state.
#[derive(Debug, Clone)]
struct MockSamplerState {
    id: u64,
    min_filter: SamplerFilter,
    mag_filter: SamplerFilter,
    address_mode: SamplerAddressMode,
}

impl MockSamplerState {
    fn new(
        min_filter: SamplerFilter,
        mag_filter: SamplerFilter,
        address_mode: SamplerAddressMode,
    ) -> Self {
        Self { id: next_id(), min_filter, mag_filter, address_mode }
    }
}

/// Mock compute pipeline state.
#[derive(Debug, Clone)]
struct MockComputePipelineState {
    id: u64,
    max_total_threads_per_threadgroup: usize,
    threadgroup_memory_length: usize,
    label: Option<String>,
}

impl MockComputePipelineState {
    fn new(max_total_threads: usize) -> Self {
        Self {
            id: next_id(),
            max_total_threads_per_threadgroup: max_total_threads,
            threadgroup_memory_length: 0,
            label: None,
        }
    }

    fn with_threadgroup_memory(mut self, length: usize) -> Self {
        self.threadgroup_memory_length = length;
        self
    }

    fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

/// 3D size (Metal MTLSize equivalent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Size3D {
    width: usize,
    height: usize,
    depth: usize,
}

impl Size3D {
    fn new(width: usize, height: usize, depth: usize) -> Self {
        Self { width, height, depth }
    }

    fn total(&self) -> usize {
        self.width * self.height * self.depth
    }
}

/// Encoder lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderState {
    Created,
    Encoding,
    Ended,
}

/// Resource tracking entry.
#[derive(Debug, Clone)]
struct ResourceTracker {
    bound_buffers: HashMap<usize, (u64, ResourceUsage)>,
    bound_textures: HashMap<usize, u64>,
    bound_samplers: HashMap<usize, u64>,
    write_set: HashSet<u64>,
    read_set: HashSet<u64>,
}

impl ResourceTracker {
    fn new() -> Self {
        Self {
            bound_buffers: HashMap::new(),
            bound_textures: HashMap::new(),
            bound_samplers: HashMap::new(),
            write_set: HashSet::new(),
            read_set: HashSet::new(),
        }
    }

    fn track_buffer(&mut self, slot: usize, buffer_id: u64, usage: ResourceUsage) {
        self.bound_buffers.insert(slot, (buffer_id, usage));
        match usage {
            ResourceUsage::Read => {
                self.read_set.insert(buffer_id);
            }
            ResourceUsage::Write => {
                self.write_set.insert(buffer_id);
            }
            ResourceUsage::ReadWrite => {
                self.read_set.insert(buffer_id);
                self.write_set.insert(buffer_id);
            }
        }
    }

    fn has_hazard(&self, resource_id: u64) -> bool {
        self.read_set.contains(&resource_id) && self.write_set.contains(&resource_id)
    }
}

/// Apple Silicon device limits.
const MAX_BUFFER_SLOTS: usize = 31;
const MAX_TEXTURE_SLOTS: usize = 128;
const MAX_SAMPLER_SLOTS: usize = 16;
const APPLE_SILICON_MAX_THREADS_PER_THREADGROUP: usize = 1024;
const APPLE_SILICON_MAX_THREADGROUP_MEMORY: usize = 32768;
const BUFFER_OFFSET_ALIGNMENT: usize = 16;

/// Mock Metal compute command encoder.
#[derive(Debug)]
struct MockComputeEncoder {
    id: u64,
    state: EncoderState,
    pipeline: Option<MockComputePipelineState>,
    dispatches: Vec<DispatchRecord>,
    resources: ResourceTracker,
    threadgroup_memory_lengths: Vec<usize>,
    total_threadgroup_memory: usize,
    label: Option<String>,
}

/// Record of a dispatch call.
#[derive(Debug, Clone)]
struct DispatchRecord {
    grid_size: Size3D,
    threadgroup_size: Size3D,
    pipeline_id: u64,
}

impl MockComputeEncoder {
    fn new() -> Self {
        Self {
            id: next_id(),
            state: EncoderState::Created,
            pipeline: None,
            dispatches: Vec::new(),
            resources: ResourceTracker::new(),
            threadgroup_memory_lengths: Vec::new(),
            total_threadgroup_memory: 0,
            label: None,
        }
    }

    fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    fn begin_encoding(&mut self) -> Result<(), EncoderError> {
        match self.state {
            EncoderState::Created => {
                self.state = EncoderState::Encoding;
                Ok(())
            }
            EncoderState::Encoding => Err(EncoderError::AlreadyEncoding),
            EncoderState::Ended => Err(EncoderError::EncoderReuseForbidden),
        }
    }

    fn end_encoding(&mut self) -> Result<(), EncoderError> {
        match self.state {
            EncoderState::Encoding => {
                self.state = EncoderState::Ended;
                Ok(())
            }
            EncoderState::Created => Err(EncoderError::NotEncoding),
            EncoderState::Ended => Err(EncoderError::AlreadyEnded),
        }
    }

    fn set_compute_pipeline_state(
        &mut self,
        pipeline: &MockComputePipelineState,
    ) -> Result<(), EncoderError> {
        self.require_encoding()?;
        self.pipeline = Some(pipeline.clone());
        Ok(())
    }

    fn set_buffer(
        &mut self,
        buffer: &MockBuffer,
        offset: usize,
        slot: usize,
    ) -> Result<(), EncoderError> {
        self.require_encoding()?;
        if slot >= MAX_BUFFER_SLOTS {
            return Err(EncoderError::BufferSlotOutOfRange { slot, max: MAX_BUFFER_SLOTS - 1 });
        }
        if offset % BUFFER_OFFSET_ALIGNMENT != 0 {
            return Err(EncoderError::BufferOffsetMisaligned {
                offset,
                alignment: BUFFER_OFFSET_ALIGNMENT,
            });
        }
        if offset > buffer.length {
            return Err(EncoderError::BufferOffsetOutOfBounds { offset, length: buffer.length });
        }
        self.resources.track_buffer(slot, buffer.id, ResourceUsage::ReadWrite);
        Ok(())
    }

    fn set_buffer_with_usage(
        &mut self,
        buffer: &MockBuffer,
        offset: usize,
        slot: usize,
        usage: ResourceUsage,
    ) -> Result<(), EncoderError> {
        self.require_encoding()?;
        if slot >= MAX_BUFFER_SLOTS {
            return Err(EncoderError::BufferSlotOutOfRange { slot, max: MAX_BUFFER_SLOTS - 1 });
        }
        if offset % BUFFER_OFFSET_ALIGNMENT != 0 {
            return Err(EncoderError::BufferOffsetMisaligned {
                offset,
                alignment: BUFFER_OFFSET_ALIGNMENT,
            });
        }
        if offset > buffer.length {
            return Err(EncoderError::BufferOffsetOutOfBounds { offset, length: buffer.length });
        }
        self.resources.track_buffer(slot, buffer.id, usage);
        Ok(())
    }

    fn set_texture(&mut self, texture: &MockTexture, slot: usize) -> Result<(), EncoderError> {
        self.require_encoding()?;
        if slot >= MAX_TEXTURE_SLOTS {
            return Err(EncoderError::TextureSlotOutOfRange { slot, max: MAX_TEXTURE_SLOTS - 1 });
        }
        self.resources.bound_textures.insert(slot, texture.id);
        Ok(())
    }

    fn set_sampler_state(
        &mut self,
        sampler: &MockSamplerState,
        slot: usize,
    ) -> Result<(), EncoderError> {
        self.require_encoding()?;
        if slot >= MAX_SAMPLER_SLOTS {
            return Err(EncoderError::SamplerSlotOutOfRange { slot, max: MAX_SAMPLER_SLOTS - 1 });
        }
        self.resources.bound_samplers.insert(slot, sampler.id);
        Ok(())
    }

    fn set_threadgroup_memory_length(
        &mut self,
        length: usize,
        index: usize,
    ) -> Result<(), EncoderError> {
        self.require_encoding()?;
        while self.threadgroup_memory_lengths.len() <= index {
            self.threadgroup_memory_lengths.push(0);
        }
        let old = self.threadgroup_memory_lengths[index];
        self.total_threadgroup_memory = self.total_threadgroup_memory - old + length;
        if self.total_threadgroup_memory > APPLE_SILICON_MAX_THREADGROUP_MEMORY {
            self.total_threadgroup_memory = self.total_threadgroup_memory - length + old;
            return Err(EncoderError::ThreadgroupMemoryExceedsMax {
                requested: self.total_threadgroup_memory - old + length,
                max: APPLE_SILICON_MAX_THREADGROUP_MEMORY,
            });
        }
        self.threadgroup_memory_lengths[index] = length;
        Ok(())
    }

    fn dispatch_threadgroups(
        &mut self,
        grid_size: Size3D,
        threadgroup_size: Size3D,
    ) -> Result<(), EncoderError> {
        self.require_encoding()?;
        let pipeline = self.pipeline.as_ref().ok_or(EncoderError::DispatchWithoutPipeline)?;

        if grid_size.width == 0 || grid_size.height == 0 || grid_size.depth == 0 {
            return Err(EncoderError::GridSizeZero);
        }
        if threadgroup_size.width == 0
            || threadgroup_size.height == 0
            || threadgroup_size.depth == 0
        {
            return Err(EncoderError::ThreadgroupDimensionZero);
        }
        let total = threadgroup_size.total();
        if total > APPLE_SILICON_MAX_THREADS_PER_THREADGROUP {
            return Err(EncoderError::ThreadgroupExceedsMax {
                requested: total,
                max: APPLE_SILICON_MAX_THREADS_PER_THREADGROUP,
            });
        }
        if total > pipeline.max_total_threads_per_threadgroup {
            return Err(EncoderError::ThreadgroupExceedsMax {
                requested: total,
                max: pipeline.max_total_threads_per_threadgroup,
            });
        }

        self.dispatches.push(DispatchRecord {
            grid_size,
            threadgroup_size,
            pipeline_id: pipeline.id,
        });
        Ok(())
    }

    fn dispatch_threads(
        &mut self,
        grid_size: Size3D,
        threadgroup_size: Size3D,
    ) -> Result<(), EncoderError> {
        // dispatch_threads uses non-uniform threadgroups (Metal 2+)
        self.require_encoding()?;
        let pipeline = self.pipeline.as_ref().ok_or(EncoderError::DispatchWithoutPipeline)?;

        if grid_size.width == 0 || grid_size.height == 0 || grid_size.depth == 0 {
            return Err(EncoderError::GridSizeZero);
        }
        if threadgroup_size.width == 0
            || threadgroup_size.height == 0
            || threadgroup_size.depth == 0
        {
            return Err(EncoderError::ThreadgroupDimensionZero);
        }
        let total = threadgroup_size.total();
        if total > APPLE_SILICON_MAX_THREADS_PER_THREADGROUP {
            return Err(EncoderError::ThreadgroupExceedsMax {
                requested: total,
                max: APPLE_SILICON_MAX_THREADS_PER_THREADGROUP,
            });
        }
        if total > pipeline.max_total_threads_per_threadgroup {
            return Err(EncoderError::ThreadgroupExceedsMax {
                requested: total,
                max: pipeline.max_total_threads_per_threadgroup,
            });
        }

        self.dispatches.push(DispatchRecord {
            grid_size,
            threadgroup_size,
            pipeline_id: pipeline.id,
        });
        Ok(())
    }

    fn require_encoding(&self) -> Result<(), EncoderError> {
        match self.state {
            EncoderState::Encoding => Ok(()),
            EncoderState::Created => Err(EncoderError::NotEncoding),
            EncoderState::Ended => Err(EncoderError::AlreadyEnded),
        }
    }
}

/// Mock command buffer that produces encoders.
struct MockCommandBuffer {
    id: u64,
    encoders_created: usize,
    committed: bool,
}

impl MockCommandBuffer {
    fn new() -> Self {
        Self { id: next_id(), encoders_created: 0, committed: false }
    }

    fn make_compute_encoder(&mut self) -> MockComputeEncoder {
        self.encoders_created += 1;
        MockComputeEncoder::new()
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

/// Helper: create a pipeline that allows up to `n` threads/threadgroup.
fn pipeline_with_max_threads(n: usize) -> MockComputePipelineState {
    MockComputePipelineState::new(n)
}

/// Helper: warp-aligned threadgroup size (Apple GPU warp = 32).
fn warp_aligned_size(threads: usize) -> usize {
    ((threads + 31) / 32) * 32
}

// ============================================================================
// 1. Encoder Lifecycle (20 tests)
// ============================================================================

#[test]
fn test_encoder_initial_state_is_created() {
    let encoder = MockComputeEncoder::new();
    assert_eq!(encoder.state, EncoderState::Created);
}

#[test]
fn test_encoder_begin_encoding_transitions_to_encoding() {
    let mut encoder = MockComputeEncoder::new();
    assert!(encoder.begin_encoding().is_ok());
    assert_eq!(encoder.state, EncoderState::Encoding);
}

#[test]
fn test_encoder_end_encoding_transitions_to_ended() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    assert!(encoder.end_encoding().is_ok());
    assert_eq!(encoder.state, EncoderState::Ended);
}

#[test]
fn test_encoder_double_begin_fails() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    assert_eq!(encoder.begin_encoding(), Err(EncoderError::AlreadyEncoding));
}

#[test]
fn test_encoder_end_without_begin_fails() {
    let mut encoder = MockComputeEncoder::new();
    assert_eq!(encoder.end_encoding(), Err(EncoderError::NotEncoding));
}

#[test]
fn test_encoder_double_end_fails() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(encoder.end_encoding(), Err(EncoderError::AlreadyEnded));
}

#[test]
fn test_encoder_begin_after_end_fails() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    encoder.end_encoding().unwrap();
    assert_eq!(encoder.begin_encoding(), Err(EncoderError::EncoderReuseForbidden));
}

#[test]
fn test_encoder_operations_require_encoding_state() {
    let mut encoder = MockComputeEncoder::new();
    let buf = MockBuffer::new(1024);
    assert_eq!(encoder.set_buffer(&buf, 0, 0), Err(EncoderError::NotEncoding));
}

#[test]
fn test_encoder_operations_after_end_fail() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    encoder.end_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    assert_eq!(encoder.set_buffer(&buf, 0, 0), Err(EncoderError::AlreadyEnded));
}

#[test]
fn test_sequential_encoders_from_command_buffer() {
    let mut cmd_buf = MockCommandBuffer::new();
    let mut enc1 = cmd_buf.make_compute_encoder();
    enc1.begin_encoding().unwrap();
    enc1.end_encoding().unwrap();

    let mut enc2 = cmd_buf.make_compute_encoder();
    enc2.begin_encoding().unwrap();
    enc2.end_encoding().unwrap();

    assert_eq!(cmd_buf.encoders_created, 2);
}

#[test]
fn test_encoder_has_unique_id() {
    let enc1 = MockComputeEncoder::new();
    let enc2 = MockComputeEncoder::new();
    assert_ne!(enc1.id, enc2.id);
}

#[test]
fn test_encoder_with_label() {
    let encoder = MockComputeEncoder::new().with_label("matmul_pass");
    assert_eq!(encoder.label.as_deref(), Some("matmul_pass"));
}

#[test]
fn test_encoder_no_dispatches_initially() {
    let encoder = MockComputeEncoder::new();
    assert!(encoder.dispatches.is_empty());
}

#[test]
fn test_encoder_tracks_dispatch_count() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(64, 1, 1)).unwrap();
    encoder.dispatch_threadgroups(Size3D::new(2, 1, 1), Size3D::new(64, 1, 1)).unwrap();
    assert_eq!(encoder.dispatches.len(), 2);
}

#[test]
fn test_command_buffer_commit() {
    let mut cmd_buf = MockCommandBuffer::new();
    assert!(!cmd_buf.committed);
    cmd_buf.commit();
    assert!(cmd_buf.committed);
}

#[test]
fn test_encoder_pipeline_initially_none() {
    let encoder = MockComputeEncoder::new();
    assert!(encoder.pipeline.is_none());
}

#[test]
fn test_encoder_set_pipeline_stores_state() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(512);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert_eq!(encoder.pipeline.as_ref().unwrap().max_total_threads_per_threadgroup, 512);
}

#[test]
fn test_encoder_resource_tracker_initially_empty() {
    let encoder = MockComputeEncoder::new();
    assert!(encoder.resources.bound_buffers.is_empty());
    assert!(encoder.resources.bound_textures.is_empty());
    assert!(encoder.resources.bound_samplers.is_empty());
}

#[test]
fn test_encoder_lifecycle_full_cycle_with_dispatch() {
    let mut cmd_buf = MockCommandBuffer::new();
    let mut encoder = cmd_buf.make_compute_encoder();
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf = MockBuffer::new(4096);
    encoder.set_buffer(&buf, 0, 0).unwrap();

    encoder.dispatch_threadgroups(Size3D::new(4, 1, 1), Size3D::new(256, 1, 1)).unwrap();

    encoder.end_encoding().unwrap();
    cmd_buf.commit();

    assert_eq!(encoder.state, EncoderState::Ended);
    assert!(cmd_buf.committed);
    assert_eq!(encoder.dispatches.len(), 1);
}

#[test]
fn test_encoder_multiple_pipeline_switches() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();

    let p1 = pipeline_with_max_threads(256);
    let p2 = pipeline_with_max_threads(512);

    encoder.set_compute_pipeline_state(&p1).unwrap();
    encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(256, 1, 1)).unwrap();

    encoder.set_compute_pipeline_state(&p2).unwrap();
    encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(512, 1, 1)).unwrap();

    assert_eq!(encoder.dispatches.len(), 2);
    assert_ne!(encoder.dispatches[0].pipeline_id, encoder.dispatches[1].pipeline_id);
}

// ============================================================================
// 2. Buffer Binding (20 tests)
// ============================================================================

#[test]
fn test_bind_buffer_slot_zero() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    assert!(encoder.set_buffer(&buf, 0, 0).is_ok());
    assert!(encoder.resources.bound_buffers.contains_key(&0));
}

#[test]
fn test_bind_buffer_max_valid_slot() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(256);
    assert!(encoder.set_buffer(&buf, 0, MAX_BUFFER_SLOTS - 1).is_ok());
}

#[test]
fn test_bind_buffer_slot_out_of_range() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(256);
    assert_eq!(
        encoder.set_buffer(&buf, 0, MAX_BUFFER_SLOTS),
        Err(EncoderError::BufferSlotOutOfRange {
            slot: MAX_BUFFER_SLOTS,
            max: MAX_BUFFER_SLOTS - 1,
        })
    );
}

#[test]
fn test_bind_buffer_large_slot_number() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(256);
    assert_eq!(
        encoder.set_buffer(&buf, 0, 100),
        Err(EncoderError::BufferSlotOutOfRange { slot: 100, max: MAX_BUFFER_SLOTS - 1 })
    );
}

#[test]
fn test_bind_buffer_with_aligned_offset() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    assert!(encoder.set_buffer(&buf, 16, 0).is_ok());
    assert!(encoder.set_buffer(&buf, 32, 1).is_ok());
    assert!(encoder.set_buffer(&buf, 256, 2).is_ok());
}

#[test]
fn test_bind_buffer_misaligned_offset() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    assert_eq!(
        encoder.set_buffer(&buf, 7, 0),
        Err(EncoderError::BufferOffsetMisaligned { offset: 7, alignment: BUFFER_OFFSET_ALIGNMENT })
    );
}

#[test]
fn test_bind_buffer_offset_one_byte_misaligned() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    assert_eq!(
        encoder.set_buffer(&buf, 1, 0),
        Err(EncoderError::BufferOffsetMisaligned { offset: 1, alignment: BUFFER_OFFSET_ALIGNMENT })
    );
}

#[test]
fn test_bind_buffer_offset_out_of_bounds() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(256);
    assert_eq!(
        encoder.set_buffer(&buf, 512, 0),
        Err(EncoderError::BufferOffsetOutOfBounds { offset: 512, length: 256 })
    );
}

#[test]
fn test_bind_buffer_offset_at_boundary() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(256);
    // offset == length is allowed (zero-length tail)
    assert!(encoder.set_buffer(&buf, 256, 0).is_ok());
}

#[test]
fn test_bind_multiple_buffers_to_different_slots() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let b0 = MockBuffer::new(1024);
    let b1 = MockBuffer::new(2048);
    let b2 = MockBuffer::new(4096);
    encoder.set_buffer(&b0, 0, 0).unwrap();
    encoder.set_buffer(&b1, 0, 1).unwrap();
    encoder.set_buffer(&b2, 0, 2).unwrap();
    assert_eq!(encoder.resources.bound_buffers.len(), 3);
}

#[test]
fn test_overwrite_buffer_in_same_slot() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let b1 = MockBuffer::new(1024);
    let b2 = MockBuffer::new(2048);
    encoder.set_buffer(&b1, 0, 0).unwrap();
    encoder.set_buffer(&b2, 0, 0).unwrap();
    let (id, _) = encoder.resources.bound_buffers[&0];
    assert_eq!(id, b2.id);
}

#[test]
fn test_bind_all_buffer_slots() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    for i in 0..MAX_BUFFER_SLOTS {
        let buf = MockBuffer::new(64);
        encoder.set_buffer(&buf, 0, i).unwrap();
    }
    assert_eq!(encoder.resources.bound_buffers.len(), MAX_BUFFER_SLOTS);
}

#[test]
fn test_bind_buffer_zero_length() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(0);
    assert!(encoder.set_buffer(&buf, 0, 0).is_ok());
}

#[test]
fn test_bind_buffer_with_label() {
    let buf = MockBuffer::new(1024).with_label("weights");
    assert_eq!(buf.label.as_deref(), Some("weights"));
    assert_eq!(buf.length, 1024);
}

#[test]
fn test_bind_buffer_with_read_usage() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::Read).unwrap();
    let (_, usage) = encoder.resources.bound_buffers[&0];
    assert_eq!(usage, ResourceUsage::Read);
}

#[test]
fn test_bind_buffer_with_write_usage() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::Write).unwrap();
    let (_, usage) = encoder.resources.bound_buffers[&0];
    assert_eq!(usage, ResourceUsage::Write);
}

#[test]
fn test_bind_buffer_with_readwrite_usage() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::ReadWrite).unwrap();
    let (_, usage) = encoder.resources.bound_buffers[&0];
    assert_eq!(usage, ResourceUsage::ReadWrite);
}

#[test]
fn test_bind_buffer_offset_multiples_of_alignment() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(4096);
    for mult in 0..16 {
        let offset = mult * BUFFER_OFFSET_ALIGNMENT;
        assert!(encoder.set_buffer(&buf, offset, 0).is_ok(), "offset {offset} should be valid");
    }
}

#[test]
fn test_bind_buffer_same_buffer_multiple_slots() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(4096);
    encoder.set_buffer(&buf, 0, 0).unwrap();
    encoder.set_buffer(&buf, 0, 1).unwrap();
    encoder.set_buffer(&buf, 0, 2).unwrap();
    assert_eq!(encoder.resources.bound_buffers.len(), 3);
    for slot in 0..3 {
        let (id, _) = encoder.resources.bound_buffers[&slot];
        assert_eq!(id, buf.id);
    }
}

#[test]
fn test_bind_buffer_with_various_misaligned_offsets() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(4096);
    for bad_offset in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let result = encoder.set_buffer(&buf, bad_offset, 0);
        assert!(
            matches!(result, Err(EncoderError::BufferOffsetMisaligned { .. })),
            "offset {bad_offset} should be misaligned"
        );
    }
}

// ============================================================================
// 3. Threadgroup Configuration (20 tests)
// ============================================================================

#[test]
fn test_dispatch_1d_threadgroups() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert!(encoder.dispatch_threadgroups(Size3D::new(16, 1, 1), Size3D::new(256, 1, 1)).is_ok());
}

#[test]
fn test_dispatch_2d_threadgroups() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert!(encoder.dispatch_threadgroups(Size3D::new(8, 8, 1), Size3D::new(16, 16, 1)).is_ok());
    assert_eq!(encoder.dispatches[0].threadgroup_size.total(), 256);
}

#[test]
fn test_dispatch_3d_threadgroups() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert!(encoder.dispatch_threadgroups(Size3D::new(4, 4, 4), Size3D::new(8, 8, 8)).is_ok());
    assert_eq!(encoder.dispatches[0].threadgroup_size.total(), 512);
}

#[test]
fn test_dispatch_max_apple_silicon_threads() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert!(encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(1024, 1, 1)).is_ok());
}

#[test]
fn test_dispatch_exceeds_apple_silicon_limit() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(2048);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert_eq!(
        encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(1025, 1, 1)),
        Err(EncoderError::ThreadgroupExceedsMax {
            requested: 1025,
            max: APPLE_SILICON_MAX_THREADS_PER_THREADGROUP,
        })
    );
}

#[test]
fn test_dispatch_exceeds_pipeline_limit() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(256);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert_eq!(
        encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(512, 1, 1)),
        Err(EncoderError::ThreadgroupExceedsMax { requested: 512, max: 256 })
    );
}

#[test]
fn test_dispatch_zero_threadgroup_width() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert_eq!(
        encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(0, 1, 1)),
        Err(EncoderError::ThreadgroupDimensionZero)
    );
}

#[test]
fn test_dispatch_zero_grid_dimension() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert_eq!(
        encoder.dispatch_threadgroups(Size3D::new(0, 1, 1), Size3D::new(64, 1, 1)),
        Err(EncoderError::GridSizeZero)
    );
}

#[test]
fn test_dispatch_without_pipeline() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    assert_eq!(
        encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(64, 1, 1)),
        Err(EncoderError::DispatchWithoutPipeline)
    );
}

#[test]
fn test_threadgroup_memory_allocation() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    assert!(encoder.set_threadgroup_memory_length(4096, 0).is_ok());
    assert_eq!(encoder.total_threadgroup_memory, 4096);
}

#[test]
fn test_threadgroup_memory_multiple_allocations() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    encoder.set_threadgroup_memory_length(4096, 0).unwrap();
    encoder.set_threadgroup_memory_length(8192, 1).unwrap();
    assert_eq!(encoder.total_threadgroup_memory, 4096 + 8192);
}

#[test]
fn test_threadgroup_memory_exceeds_max() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    assert_eq!(
        encoder.set_threadgroup_memory_length(APPLE_SILICON_MAX_THREADGROUP_MEMORY + 1, 0),
        Err(EncoderError::ThreadgroupMemoryExceedsMax {
            requested: APPLE_SILICON_MAX_THREADGROUP_MEMORY + 1,
            max: APPLE_SILICON_MAX_THREADGROUP_MEMORY,
        })
    );
}

#[test]
fn test_threadgroup_memory_exactly_at_max() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    assert!(encoder.set_threadgroup_memory_length(APPLE_SILICON_MAX_THREADGROUP_MEMORY, 0).is_ok());
}

#[test]
fn test_warp_aligned_threadgroup_sizes() {
    assert_eq!(warp_aligned_size(1), 32);
    assert_eq!(warp_aligned_size(32), 32);
    assert_eq!(warp_aligned_size(33), 64);
    assert_eq!(warp_aligned_size(64), 64);
    assert_eq!(warp_aligned_size(100), 128);
    assert_eq!(warp_aligned_size(1024), 1024);
}

#[test]
fn test_dispatch_threads_non_uniform() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    // dispatch_threads allows non-multiple grid sizes
    assert!(encoder.dispatch_threads(Size3D::new(1000, 1, 1), Size3D::new(256, 1, 1)).is_ok());
}

#[test]
fn test_dispatch_2d_threadgroup_product_limit() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    // 32*32 = 1024 ≤ limit
    assert!(encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(32, 32, 1)).is_ok());
    // 33*32 = 1056 > 1024
    assert!(encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(33, 32, 1)).is_err());
}

#[test]
fn test_dispatch_3d_threadgroup_product_limit() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    // 8*8*16 = 1024
    assert!(encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(8, 8, 16)).is_ok());
    // 8*8*17 = 1088 > 1024
    assert!(encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(8, 8, 17)).is_err());
}

#[test]
fn test_dispatch_records_grid_size() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    let grid = Size3D::new(10, 20, 30);
    let tg = Size3D::new(8, 4, 2);
    encoder.dispatch_threadgroups(grid, tg).unwrap();
    assert_eq!(encoder.dispatches[0].grid_size, grid);
    assert_eq!(encoder.dispatches[0].threadgroup_size, tg);
}

#[test]
fn test_threadgroup_memory_replace_at_index() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    encoder.set_threadgroup_memory_length(1024, 0).unwrap();
    assert_eq!(encoder.total_threadgroup_memory, 1024);
    encoder.set_threadgroup_memory_length(2048, 0).unwrap();
    assert_eq!(encoder.total_threadgroup_memory, 2048);
}

#[test]
fn test_dispatch_zero_grid_depth() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    assert_eq!(
        encoder.dispatch_threadgroups(Size3D::new(1, 1, 0), Size3D::new(64, 1, 1)),
        Err(EncoderError::GridSizeZero)
    );
}

// ============================================================================
// 4. Resource Validation (15 tests)
// ============================================================================

#[test]
fn test_bind_texture_1d() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let tex = MockTexture::new(TextureType::Texture1D, PixelFormat::R32Float, 256);
    assert!(encoder.set_texture(&tex, 0).is_ok());
    assert!(encoder.resources.bound_textures.contains_key(&0));
}

#[test]
fn test_bind_texture_2d() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let tex = MockTexture::new(TextureType::Texture2D, PixelFormat::RGBA8Unorm, 512)
        .with_size(512, 512, 1);
    assert!(encoder.set_texture(&tex, 0).is_ok());
}

#[test]
fn test_bind_texture_3d() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let tex =
        MockTexture::new(TextureType::Texture3D, PixelFormat::R16Float, 64).with_size(64, 64, 64);
    assert!(encoder.set_texture(&tex, 0).is_ok());
}

#[test]
fn test_texture_slot_out_of_range() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let tex = MockTexture::new(TextureType::Texture2D, PixelFormat::R32Float, 64);
    assert_eq!(
        encoder.set_texture(&tex, MAX_TEXTURE_SLOTS),
        Err(EncoderError::TextureSlotOutOfRange {
            slot: MAX_TEXTURE_SLOTS,
            max: MAX_TEXTURE_SLOTS - 1,
        })
    );
}

#[test]
fn test_bind_sampler_state() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let sampler = MockSamplerState::new(
        SamplerFilter::Linear,
        SamplerFilter::Linear,
        SamplerAddressMode::ClampToEdge,
    );
    assert!(encoder.set_sampler_state(&sampler, 0).is_ok());
    assert!(encoder.resources.bound_samplers.contains_key(&0));
}

#[test]
fn test_sampler_slot_out_of_range() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let sampler = MockSamplerState::new(
        SamplerFilter::Nearest,
        SamplerFilter::Nearest,
        SamplerAddressMode::Repeat,
    );
    assert_eq!(
        encoder.set_sampler_state(&sampler, MAX_SAMPLER_SLOTS),
        Err(EncoderError::SamplerSlotOutOfRange {
            slot: MAX_SAMPLER_SLOTS,
            max: MAX_SAMPLER_SLOTS - 1,
        })
    );
}

#[test]
fn test_resource_usage_tracking_read() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::Read).unwrap();
    assert!(encoder.resources.read_set.contains(&buf.id));
    assert!(!encoder.resources.write_set.contains(&buf.id));
}

#[test]
fn test_resource_usage_tracking_write() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::Write).unwrap();
    assert!(!encoder.resources.read_set.contains(&buf.id));
    assert!(encoder.resources.write_set.contains(&buf.id));
}

#[test]
fn test_resource_usage_tracking_readwrite() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::ReadWrite).unwrap();
    assert!(encoder.resources.read_set.contains(&buf.id));
    assert!(encoder.resources.write_set.contains(&buf.id));
}

#[test]
fn test_resource_hazard_detection_read_write_same() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf, 0, 1, ResourceUsage::Write).unwrap();
    assert!(encoder.resources.has_hazard(buf.id));
}

#[test]
fn test_resource_no_hazard_read_only() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let buf = MockBuffer::new(1024);
    encoder.set_buffer_with_usage(&buf, 0, 0, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf, 0, 1, ResourceUsage::Read).unwrap();
    assert!(!encoder.resources.has_hazard(buf.id));
}

#[test]
fn test_resource_lifetime_across_dispatches() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf = MockBuffer::new(4096);
    encoder.set_buffer(&buf, 0, 0).unwrap();
    encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(64, 1, 1)).unwrap();
    // Buffer remains bound for next dispatch
    assert!(encoder.resources.bound_buffers.contains_key(&0));
    encoder.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(64, 1, 1)).unwrap();
    assert_eq!(encoder.dispatches.len(), 2);
}

#[test]
fn test_bind_max_textures() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    for i in 0..MAX_TEXTURE_SLOTS {
        let tex = MockTexture::new(TextureType::Texture2D, PixelFormat::R32Float, 64);
        encoder.set_texture(&tex, i).unwrap();
    }
    assert_eq!(encoder.resources.bound_textures.len(), MAX_TEXTURE_SLOTS);
}

#[test]
fn test_bind_max_samplers() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    for i in 0..MAX_SAMPLER_SLOTS {
        let sampler = MockSamplerState::new(
            SamplerFilter::Linear,
            SamplerFilter::Linear,
            SamplerAddressMode::ClampToEdge,
        );
        encoder.set_sampler_state(&sampler, i).unwrap();
    }
    assert_eq!(encoder.resources.bound_samplers.len(), MAX_SAMPLER_SLOTS);
}

#[test]
fn test_texture_overwrite_in_same_slot() {
    let mut encoder = MockComputeEncoder::new();
    encoder.begin_encoding().unwrap();
    let t1 = MockTexture::new(TextureType::Texture2D, PixelFormat::R32Float, 64);
    let t2 = MockTexture::new(TextureType::Texture2D, PixelFormat::RGBA16Float, 128);
    encoder.set_texture(&t1, 0).unwrap();
    encoder.set_texture(&t2, 0).unwrap();
    assert_eq!(encoder.resources.bound_textures[&0], t2.id);
}

// ============================================================================
// 5. Error Handling (15 tests)
// ============================================================================

#[test]
fn test_error_display_not_encoding() {
    let err = EncoderError::NotEncoding;
    assert_eq!(format!("{err}"), "encoder is not in encoding state");
}

#[test]
fn test_error_display_already_encoding() {
    let err = EncoderError::AlreadyEncoding;
    assert_eq!(format!("{err}"), "encoder is already in encoding state");
}

#[test]
fn test_error_display_already_ended() {
    let err = EncoderError::AlreadyEnded;
    assert_eq!(format!("{err}"), "encoder has already ended");
}

#[test]
fn test_error_display_buffer_slot() {
    let err = EncoderError::BufferSlotOutOfRange { slot: 50, max: 30 };
    assert_eq!(format!("{err}"), "buffer slot 50 exceeds max 30");
}

#[test]
fn test_error_display_buffer_offset_misaligned() {
    let err = EncoderError::BufferOffsetMisaligned { offset: 7, alignment: 16 };
    assert_eq!(format!("{err}"), "buffer offset 7 not aligned to 16");
}

#[test]
fn test_error_display_threadgroup_exceeds() {
    let err = EncoderError::ThreadgroupExceedsMax { requested: 2048, max: 1024 };
    assert_eq!(format!("{err}"), "threadgroup size 2048 exceeds max 1024");
}

#[test]
fn test_error_display_threadgroup_memory() {
    let err = EncoderError::ThreadgroupMemoryExceedsMax { requested: 65536, max: 32768 };
    assert_eq!(format!("{err}"), "threadgroup memory 65536 bytes exceeds max 32768 bytes");
}

#[test]
fn test_error_display_no_pipeline() {
    let err = EncoderError::NoPipelineSet;
    assert_eq!(format!("{err}"), "no compute pipeline state set");
}

#[test]
fn test_error_display_resource_hazard() {
    let err = EncoderError::ResourceHazard { resource_id: 42 };
    assert_eq!(format!("{err}"), "resource hazard on resource 42");
}

#[test]
fn test_error_display_encoder_reuse() {
    let err = EncoderError::EncoderReuseForbidden;
    assert_eq!(format!("{err}"), "encoder cannot be reused after ending");
}

#[test]
fn test_error_display_grid_size_zero() {
    let err = EncoderError::GridSizeZero;
    assert_eq!(format!("{err}"), "grid size must be non-zero in all dimensions");
}

#[test]
fn test_error_display_dispatch_without_pipeline() {
    let err = EncoderError::DispatchWithoutPipeline;
    assert_eq!(format!("{err}"), "dispatch called without a pipeline state set");
}

#[test]
fn test_error_equality() {
    assert_eq!(EncoderError::NotEncoding, EncoderError::NotEncoding);
    assert_ne!(EncoderError::NotEncoding, EncoderError::AlreadyEncoding);
}

#[test]
fn test_recovery_new_encoder_after_error() {
    let mut cmd_buf = MockCommandBuffer::new();

    // First encoder encounters error
    let mut enc1 = cmd_buf.make_compute_encoder();
    enc1.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(64);
    enc1.set_compute_pipeline_state(&pipeline).unwrap();
    // Dispatch fails (too many threads)
    let result = enc1.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(128, 1, 1));
    assert!(result.is_err());
    enc1.end_encoding().unwrap();

    // Second encoder works fine
    let mut enc2 = cmd_buf.make_compute_encoder();
    enc2.begin_encoding().unwrap();
    enc2.set_compute_pipeline_state(&pipeline).unwrap();
    assert!(enc2.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(64, 1, 1)).is_ok());
    enc2.end_encoding().unwrap();
}

#[test]
fn test_error_display_nested_encoding() {
    let err = EncoderError::NestedEncodingForbidden;
    assert_eq!(format!("{err}"), "nested encoding passes are forbidden");
}

// ============================================================================
// 6. Integration Scenarios (10 tests)
// ============================================================================

#[test]
fn test_matmul_dispatch_pattern() {
    // Matrix multiply: C[M,N] = A[M,K] * B[K,N]
    let m = 1024_usize;
    let n = 1024_usize;
    let k = 512_usize;

    let mut encoder = MockComputeEncoder::new().with_label("matmul");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024).with_label("matmul_f32");
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_a = MockBuffer::new(m * k * 4);
    let buf_b = MockBuffer::new(k * n * 4);
    let buf_c = MockBuffer::new(m * n * 4);

    encoder.set_buffer_with_usage(&buf_a, 0, 0, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_b, 0, 1, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_c, 0, 2, ResourceUsage::Write).unwrap();

    let tile_m = 16;
    let tile_n = 16;
    let grid = Size3D::new((n + tile_n - 1) / tile_n, (m + tile_m - 1) / tile_m, 1);
    let tg = Size3D::new(tile_n, tile_m, 1);

    encoder.dispatch_threadgroups(grid, tg).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.dispatches.len(), 1);
    assert_eq!(encoder.dispatches[0].grid_size, Size3D::new(64, 64, 1));
    assert_eq!(encoder.dispatches[0].threadgroup_size, Size3D::new(16, 16, 1));
}

#[test]
fn test_softmax_dispatch_pattern() {
    // Softmax over rows: input[batch, seq_len], output[batch, seq_len]
    let batch = 32_usize;
    let seq_len = 128_usize;

    let mut encoder = MockComputeEncoder::new().with_label("softmax");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024).with_label("softmax_f32");
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_in = MockBuffer::new(batch * seq_len * 4);
    let buf_out = MockBuffer::new(batch * seq_len * 4);

    encoder.set_buffer_with_usage(&buf_in, 0, 0, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_out, 0, 1, ResourceUsage::Write).unwrap();

    // Threadgroup shared memory for row max and sum
    encoder.set_threadgroup_memory_length(warp_aligned_size(seq_len) * 4, 0).unwrap();

    // One threadgroup per batch row
    let tg_size = warp_aligned_size(seq_len);
    encoder.dispatch_threadgroups(Size3D::new(batch, 1, 1), Size3D::new(tg_size, 1, 1)).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.dispatches.len(), 1);
    assert_eq!(encoder.dispatches[0].grid_size.width, 32);
}

#[test]
fn test_elementwise_add_dispatch() {
    let n = 65536_usize;

    let mut encoder = MockComputeEncoder::new().with_label("elementwise_add");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_a = MockBuffer::new(n * 4);
    let buf_b = MockBuffer::new(n * 4);
    let buf_c = MockBuffer::new(n * 4);

    encoder.set_buffer(&buf_a, 0, 0).unwrap();
    encoder.set_buffer(&buf_b, 0, 1).unwrap();
    encoder.set_buffer(&buf_c, 0, 2).unwrap();

    let tg_size = 256_usize;
    let grid_x = (n + tg_size - 1) / tg_size;
    encoder.dispatch_threadgroups(Size3D::new(grid_x, 1, 1), Size3D::new(tg_size, 1, 1)).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.dispatches[0].grid_size.width, 256);
    assert_eq!(encoder.dispatches[0].threadgroup_size.width, 256);
}

#[test]
fn test_reduction_sum_dispatch() {
    // Two-pass reduction: first pass partial sums, second pass final sum
    let n = 1024 * 1024_usize;

    let mut cmd_buf = MockCommandBuffer::new();

    // Pass 1: reduce chunks
    let tg_size = 256_usize;
    let num_groups = (n + tg_size - 1) / tg_size;

    let mut enc1 = cmd_buf.make_compute_encoder();
    enc1.begin_encoding().unwrap();
    let pipeline = pipeline_with_max_threads(1024).with_label("reduce_sum_pass1");
    enc1.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_in = MockBuffer::new(n * 4);
    let buf_partial = MockBuffer::new(num_groups * 4);
    enc1.set_buffer_with_usage(&buf_in, 0, 0, ResourceUsage::Read).unwrap();
    enc1.set_buffer_with_usage(&buf_partial, 0, 1, ResourceUsage::Write).unwrap();
    enc1.set_threadgroup_memory_length(tg_size * 4, 0).unwrap();
    enc1.dispatch_threadgroups(Size3D::new(num_groups, 1, 1), Size3D::new(tg_size, 1, 1)).unwrap();
    enc1.end_encoding().unwrap();

    // Pass 2: final reduction
    let mut enc2 = cmd_buf.make_compute_encoder();
    enc2.begin_encoding().unwrap();
    let pipeline2 = pipeline_with_max_threads(1024).with_label("reduce_sum_pass2");
    enc2.set_compute_pipeline_state(&pipeline2).unwrap();
    let buf_out = MockBuffer::new(4);
    enc2.set_buffer_with_usage(&buf_partial, 0, 0, ResourceUsage::Read).unwrap();
    enc2.set_buffer_with_usage(&buf_out, 0, 1, ResourceUsage::Write).unwrap();
    let pass2_tg = warp_aligned_size(num_groups.min(1024));
    enc2.set_threadgroup_memory_length(pass2_tg * 4, 0).unwrap();
    enc2.dispatch_threadgroups(Size3D::new(1, 1, 1), Size3D::new(pass2_tg, 1, 1)).unwrap();
    enc2.end_encoding().unwrap();

    cmd_buf.commit();

    assert_eq!(cmd_buf.encoders_created, 2);
    assert!(cmd_buf.committed);
}

#[test]
fn test_attention_compute_dispatch() {
    // Self-attention: Q*K^T/sqrt(d) then softmax then *V
    let batch = 1_usize;
    let heads = 8_usize;
    let seq_len = 128_usize;
    let head_dim = 64_usize;

    let mut cmd_buf = MockCommandBuffer::new();

    // Step 1: QK^T
    let mut enc1 = cmd_buf.make_compute_encoder();
    enc1.begin_encoding().unwrap();
    let p_qkt = pipeline_with_max_threads(1024).with_label("qk_transpose");
    enc1.set_compute_pipeline_state(&p_qkt).unwrap();

    let buf_q = MockBuffer::new(batch * heads * seq_len * head_dim * 4);
    let buf_k = MockBuffer::new(batch * heads * seq_len * head_dim * 4);
    let buf_scores = MockBuffer::new(batch * heads * seq_len * seq_len * 4);

    enc1.set_buffer_with_usage(&buf_q, 0, 0, ResourceUsage::Read).unwrap();
    enc1.set_buffer_with_usage(&buf_k, 0, 1, ResourceUsage::Read).unwrap();
    enc1.set_buffer_with_usage(&buf_scores, 0, 2, ResourceUsage::Write).unwrap();

    let tg = Size3D::new(16, 16, 1);
    let grid = Size3D::new((seq_len + 15) / 16, (seq_len + 15) / 16, batch * heads);
    enc1.dispatch_threadgroups(grid, tg).unwrap();
    enc1.end_encoding().unwrap();

    // Step 2: Softmax (attention weights)
    let mut enc2 = cmd_buf.make_compute_encoder();
    enc2.begin_encoding().unwrap();
    let p_sm = pipeline_with_max_threads(1024).with_label("softmax_attn");
    enc2.set_compute_pipeline_state(&p_sm).unwrap();
    enc2.set_buffer_with_usage(&buf_scores, 0, 0, ResourceUsage::ReadWrite).unwrap();
    enc2.set_threadgroup_memory_length(seq_len * 4, 0).unwrap();
    enc2.dispatch_threadgroups(
        Size3D::new(batch * heads * seq_len, 1, 1),
        Size3D::new(warp_aligned_size(seq_len), 1, 1),
    )
    .unwrap();
    enc2.end_encoding().unwrap();

    // Step 3: scores * V
    let mut enc3 = cmd_buf.make_compute_encoder();
    enc3.begin_encoding().unwrap();
    let p_sv = pipeline_with_max_threads(1024).with_label("score_v");
    enc3.set_compute_pipeline_state(&p_sv).unwrap();

    let buf_v = MockBuffer::new(batch * heads * seq_len * head_dim * 4);
    let buf_out = MockBuffer::new(batch * heads * seq_len * head_dim * 4);

    enc3.set_buffer_with_usage(&buf_scores, 0, 0, ResourceUsage::Read).unwrap();
    enc3.set_buffer_with_usage(&buf_v, 0, 1, ResourceUsage::Read).unwrap();
    enc3.set_buffer_with_usage(&buf_out, 0, 2, ResourceUsage::Write).unwrap();
    enc3.dispatch_threadgroups(
        Size3D::new((head_dim + 15) / 16, (seq_len + 15) / 16, batch * heads),
        Size3D::new(16, 16, 1),
    )
    .unwrap();
    enc3.end_encoding().unwrap();

    cmd_buf.commit();
    assert_eq!(cmd_buf.encoders_created, 3);
}

#[test]
fn test_layernorm_dispatch_pattern() {
    let batch = 16_usize;
    let hidden = 768_usize;

    let mut encoder = MockComputeEncoder::new().with_label("layernorm");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024).with_label("layernorm_f32");
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_in = MockBuffer::new(batch * hidden * 4);
    let buf_out = MockBuffer::new(batch * hidden * 4);
    let buf_gamma = MockBuffer::new(hidden * 4);
    let buf_beta = MockBuffer::new(hidden * 4);

    encoder.set_buffer_with_usage(&buf_in, 0, 0, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_out, 0, 1, ResourceUsage::Write).unwrap();
    encoder.set_buffer_with_usage(&buf_gamma, 0, 2, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_beta, 0, 3, ResourceUsage::Read).unwrap();

    // Shared memory for mean and variance
    encoder.set_threadgroup_memory_length(256 * 4 * 2, 0).unwrap();

    let tg_size = warp_aligned_size(hidden.min(1024));
    encoder.dispatch_threadgroups(Size3D::new(batch, 1, 1), Size3D::new(tg_size, 1, 1)).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.dispatches.len(), 1);
    assert_eq!(encoder.resources.bound_buffers.len(), 4);
}

#[test]
fn test_gelu_activation_dispatch() {
    let n = 32768_usize;

    let mut encoder = MockComputeEncoder::new().with_label("gelu");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_inout = MockBuffer::new(n * 4);
    encoder.set_buffer_with_usage(&buf_inout, 0, 0, ResourceUsage::ReadWrite).unwrap();

    let tg_size = 256_usize;
    let grid_x = (n + tg_size - 1) / tg_size;
    encoder.dispatch_threadgroups(Size3D::new(grid_x, 1, 1), Size3D::new(tg_size, 1, 1)).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.dispatches[0].grid_size.width, 128);
}

#[test]
fn test_quantized_matmul_i2s_dispatch() {
    // I2_S quantized matmul: 2-bit weights packed 16 per u32
    let m = 256_usize;
    let n = 512_usize;
    let k = 1024_usize;

    let mut encoder = MockComputeEncoder::new().with_label("i2s_matmul");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024).with_label("i2s_gemv_f32");
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    // Packed weights: k values * 2 bits / 8 = k/4 bytes per row
    let buf_weights = MockBuffer::new(n * (k / 4));
    let buf_scales = MockBuffer::new(n * 4); // one f32 scale per output row
    let buf_input = MockBuffer::new(m * k * 4);
    let buf_output = MockBuffer::new(m * n * 4);

    encoder.set_buffer_with_usage(&buf_weights, 0, 0, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_scales, 0, 1, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_input, 0, 2, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_output, 0, 3, ResourceUsage::Write).unwrap();

    // Shared mem for partial dot products
    encoder.set_threadgroup_memory_length(256 * 4, 0).unwrap();

    let tg = Size3D::new(256, 1, 1);
    let grid = Size3D::new(n, m, 1);
    encoder.dispatch_threadgroups(grid, tg).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.resources.bound_buffers.len(), 4);
    assert_eq!(encoder.dispatches.len(), 1);
}

#[test]
fn test_rope_embedding_dispatch() {
    // RoPE (Rotary Position Embedding)
    let batch = 1_usize;
    let seq_len = 128_usize;
    let heads = 32_usize;
    let head_dim = 64_usize;

    let mut encoder = MockComputeEncoder::new().with_label("rope");
    encoder.begin_encoding().unwrap();

    let pipeline = pipeline_with_max_threads(1024);
    encoder.set_compute_pipeline_state(&pipeline).unwrap();

    let buf_q = MockBuffer::new(batch * seq_len * heads * head_dim * 4);
    let buf_k = MockBuffer::new(batch * seq_len * heads * head_dim * 4);
    let buf_cos = MockBuffer::new(seq_len * head_dim * 4);
    let buf_sin = MockBuffer::new(seq_len * head_dim * 4);

    encoder.set_buffer_with_usage(&buf_q, 0, 0, ResourceUsage::ReadWrite).unwrap();
    encoder.set_buffer_with_usage(&buf_k, 0, 1, ResourceUsage::ReadWrite).unwrap();
    encoder.set_buffer_with_usage(&buf_cos, 0, 2, ResourceUsage::Read).unwrap();
    encoder.set_buffer_with_usage(&buf_sin, 0, 3, ResourceUsage::Read).unwrap();

    // Each threadgroup handles one (seq_pos, head) pair
    let tg_size = warp_aligned_size(head_dim / 2); // process pairs
    let grid = Size3D::new(seq_len, heads, batch);
    encoder.dispatch_threadgroups(grid, Size3D::new(tg_size, 1, 1)).unwrap();
    encoder.end_encoding().unwrap();

    assert_eq!(encoder.dispatches.len(), 1);
    assert_eq!(encoder.dispatches[0].grid_size, Size3D::new(128, 32, 1));
}

#[test]
fn test_multi_pass_pipeline_with_barriers() {
    // Simulates pass1 -> pass2 requiring barrier (separate encoders)
    let n = 4096_usize;
    let mut cmd_buf = MockCommandBuffer::new();

    // Pass 1: elementwise activation
    let mut enc1 = cmd_buf.make_compute_encoder();
    enc1.begin_encoding().unwrap();
    let p1 = pipeline_with_max_threads(1024);
    enc1.set_compute_pipeline_state(&p1).unwrap();
    let shared_buf = MockBuffer::new(n * 4);
    enc1.set_buffer_with_usage(&shared_buf, 0, 0, ResourceUsage::Write).unwrap();
    enc1.dispatch_threadgroups(Size3D::new(n / 256, 1, 1), Size3D::new(256, 1, 1)).unwrap();
    enc1.end_encoding().unwrap();

    // Pass 2: reads output of pass 1
    let mut enc2 = cmd_buf.make_compute_encoder();
    enc2.begin_encoding().unwrap();
    let p2 = pipeline_with_max_threads(1024);
    enc2.set_compute_pipeline_state(&p2).unwrap();
    enc2.set_buffer_with_usage(&shared_buf, 0, 0, ResourceUsage::Read).unwrap();
    let out_buf = MockBuffer::new(n * 4);
    enc2.set_buffer_with_usage(&out_buf, 0, 1, ResourceUsage::Write).unwrap();
    enc2.dispatch_threadgroups(Size3D::new(n / 256, 1, 1), Size3D::new(256, 1, 1)).unwrap();
    enc2.end_encoding().unwrap();

    cmd_buf.commit();
    assert_eq!(cmd_buf.encoders_created, 2);
    assert!(cmd_buf.committed);
    // No hazard on enc2 since shared_buf is read-only there
    assert!(!enc2.resources.has_hazard(shared_buf.id));
}
