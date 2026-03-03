#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal buffer management validation tests for Apple Silicon GPUs.
//!
//! Validates buffer creation, alignment, pooling, double buffering,
//! CPU↔GPU transfers, offset access, memory pressure handling, hazard
//! tracking, argument buffer encoding, size limits, debug labels, and
//! ring-buffer allocation against the Metal compute configuration layer
//! exposed by `bitnet_kernels::metal_compute`.

#![cfg(feature = "metal")]

use bitnet_kernels::metal_compute::{
    DispatchDimensions, METAL_BUFFER_ALIGNMENT, METAL_MAX_WORKGROUP_SIZE, MemoryArchitecture,
    MetalComputePipeline, MetalConfigError, WorkgroupSize, align_buffer_size, is_aligned,
};

// ── Metal-specific constants ────────────────────────────────────────

/// Metal page size on Apple Silicon (16 KiB).
const PAGE_SIZE: usize = 16384;

/// Apple Silicon SIMD group width.
const SIMD_GROUP_WIDTH: u32 = 32;

/// 4 GiB practical buffer limit on older Metal devices.
const METAL_4GB_LIMIT: usize = 4 * 1024 * 1024 * 1024;

/// Typical f32 element byte width.
const F32_BYTES: usize = 4;

/// Typical f16 element byte width.
const F16_BYTES: usize = 2;

// ── Helpers ─────────────────────────────────────────────────────────

/// Round `size` up to the next multiple of `METAL_BUFFER_ALIGNMENT`.
fn metal_align(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

/// Simulate a Metal storage mode as an enum for test purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    Shared,
    Private,
    Managed,
}

/// Simulated buffer descriptor for test validation.
#[derive(Debug, Clone)]
struct BufferDescriptor {
    label: String,
    size: usize,
    aligned_size: usize,
    storage_mode: StorageMode,
}

impl BufferDescriptor {
    fn new(label: &str, size: usize, mode: StorageMode) -> Self {
        Self { label: label.to_string(), size, aligned_size: metal_align(size), storage_mode: mode }
    }
}

/// Simple buffer pool that tracks allocations by size class.
struct BufferPool {
    pool: Vec<(usize, bool)>, // (aligned_size, in_use)
    capacity: usize,
    allocated_bytes: usize,
}

impl BufferPool {
    fn new(capacity: usize) -> Self {
        Self { pool: Vec::new(), capacity, allocated_bytes: 0 }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        let aligned = metal_align(size);
        // Try reuse first
        for (i, (s, in_use)) in self.pool.iter_mut().enumerate() {
            if !*in_use && *s >= aligned {
                *in_use = true;
                return Some(i);
            }
        }
        // New allocation
        if self.allocated_bytes + aligned > self.capacity {
            return None;
        }
        let idx = self.pool.len();
        self.pool.push((aligned, true));
        self.allocated_bytes += aligned;
        Some(idx)
    }

    fn release(&mut self, index: usize) {
        if index < self.pool.len() {
            self.pool[index].1 = false;
        }
    }

    fn in_use_count(&self) -> usize {
        self.pool.iter().filter(|(_, u)| *u).count()
    }

    fn total_allocated(&self) -> usize {
        self.allocated_bytes
    }

    fn fragmentation_ratio(&self) -> f64 {
        let free: usize = self.pool.iter().filter(|(_, u)| !*u).map(|(s, _)| *s).sum();
        if self.allocated_bytes == 0 {
            return 0.0;
        }
        free as f64 / self.allocated_bytes as f64
    }
}

/// Double-buffer (ping-pong) state.
struct DoubleBuffer {
    buffers: [Vec<u8>; 2],
    current: usize,
    flight: [bool; 2],
}

impl DoubleBuffer {
    fn new(size: usize) -> Self {
        let aligned = metal_align(size);
        Self {
            buffers: [vec![0u8; aligned], vec![0u8; aligned]],
            current: 0,
            flight: [false, false],
        }
    }

    fn current_buffer(&self) -> &[u8] {
        &self.buffers[self.current]
    }

    fn swap(&mut self) {
        self.flight[self.current] = true;
        self.current = 1 - self.current;
    }

    fn complete_flight(&mut self, index: usize) {
        self.flight[index] = false;
    }

    fn is_in_flight(&self, index: usize) -> bool {
        self.flight[index]
    }
}

/// Ring buffer for circular frame-based allocation.
struct RingBuffer {
    data: Vec<u8>,
    capacity: usize,
    head: usize,
    tail: usize,
    frame_ends: Vec<usize>,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        let aligned_cap = metal_align(capacity);
        Self {
            data: vec![0u8; aligned_cap],
            capacity: aligned_cap,
            head: 0,
            tail: 0,
            frame_ends: Vec::new(),
        }
    }

    fn allocate(&mut self, size: usize) -> Option<usize> {
        let aligned = metal_align(size);
        let available = if self.head >= self.tail {
            self.capacity - self.head + self.tail
        } else {
            self.tail - self.head
        };
        if aligned > available {
            return None;
        }
        let offset = self.head;
        self.head = (self.head + aligned) % self.capacity;
        Some(offset)
    }

    fn end_frame(&mut self) {
        self.frame_ends.push(self.head);
    }

    fn retire_frame(&mut self) {
        if let Some(end) = self.frame_ends.first().copied() {
            self.tail = end;
            self.frame_ends.remove(0);
        }
    }

    fn used_bytes(&self) -> usize {
        if self.head >= self.tail {
            self.head - self.tail
        } else {
            self.capacity - self.tail + self.head
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 1. Buffer creation (shared, private, managed storage modes)
// ═════════════════════════════════════════════════════════════════════

mod buffer_creation {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn shared_storage_mode_descriptor() {
        let desc = BufferDescriptor::new("weights", 1024, StorageMode::Shared);
        assert_eq!(desc.storage_mode, StorageMode::Shared);
        assert_eq!(desc.size, 1024);
        assert_eq!(desc.aligned_size, 1024);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn private_storage_mode_descriptor() {
        let desc = BufferDescriptor::new("intermediate", 512, StorageMode::Private);
        assert_eq!(desc.storage_mode, StorageMode::Private);
        assert_eq!(desc.aligned_size, 512);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn managed_storage_mode_descriptor() {
        let desc = BufferDescriptor::new("managed_buf", 300, StorageMode::Managed);
        assert_eq!(desc.storage_mode, StorageMode::Managed);
        assert_eq!(desc.aligned_size, 512);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn various_buffer_sizes() {
        for &size in &[1, 255, 256, 257, 1024, 4096, 65536, 1_000_000] {
            let desc = BufferDescriptor::new("test", size, StorageMode::Shared);
            assert!(
                desc.aligned_size >= size,
                "aligned_size {} must be >= original size {}",
                desc.aligned_size,
                size
            );
            assert!(
                is_aligned(desc.aligned_size),
                "aligned_size {} must be 256-byte aligned",
                desc.aligned_size
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn zero_length_buffer() {
        let desc = BufferDescriptor::new("empty", 0, StorageMode::Shared);
        assert_eq!(desc.size, 0);
        assert_eq!(desc.aligned_size, 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn buffer_creation_preserves_label() {
        let desc = BufferDescriptor::new("my_kernel_weights", 1024, StorageMode::Private);
        assert_eq!(desc.label, "my_kernel_weights");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_aligned_buffer_bytes_for_f32() {
        let p = MetalComputePipeline::new("f32_buf");
        // 100 f32 = 400 bytes → 512
        assert_eq!(p.aligned_buffer_bytes(100, F32_BYTES), 512);
        // 64 f32 = 256 bytes → 256
        assert_eq!(p.aligned_buffer_bytes(64, F32_BYTES), 256);
        // 1 f32 = 4 bytes → 256
        assert_eq!(p.aligned_buffer_bytes(1, F32_BYTES), 256);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_aligned_buffer_bytes_for_f16() {
        let p = MetalComputePipeline::new("f16_buf");
        // 128 f16 = 256 bytes → 256
        assert_eq!(p.aligned_buffer_bytes(128, F16_BYTES), 256);
        // 129 f16 = 258 bytes → 512
        assert_eq!(p.aligned_buffer_bytes(129, F16_BYTES), 512);
    }
}

// ═════════════════════════════════════════════════════════════════════
// 2. Buffer alignment (256-byte alignment for Metal)
// ═════════════════════════════════════════════════════════════════════

mod buffer_alignment {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn alignment_constant_is_256() {
        assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
        assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn align_zero_returns_zero() {
        assert_eq!(align_buffer_size(0), 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn align_exact_multiples_unchanged() {
        for mult in 1..=16 {
            let size = 256 * mult;
            assert_eq!(
                align_buffer_size(size),
                size,
                "exact multiple {size} should remain unchanged"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn align_rounds_up_non_multiples() {
        assert_eq!(align_buffer_size(1), 256);
        assert_eq!(align_buffer_size(128), 256);
        assert_eq!(align_buffer_size(255), 256);
        assert_eq!(align_buffer_size(257), 512);
        assert_eq!(align_buffer_size(511), 512);
        assert_eq!(align_buffer_size(513), 768);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn is_aligned_detects_valid_offsets() {
        assert!(is_aligned(0));
        assert!(is_aligned(256));
        assert!(is_aligned(512));
        assert!(is_aligned(1024));
        assert!(is_aligned(256 * 100));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn is_aligned_rejects_non_aligned() {
        assert!(!is_aligned(1));
        assert!(!is_aligned(127));
        assert!(!is_aligned(128));
        assert!(!is_aligned(255));
        assert!(!is_aligned(257));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn non_aligned_size_fallback_to_next_boundary() {
        // Simulates a non-aligned allocation being rounded up
        let raw_size = 1000;
        let aligned = align_buffer_size(raw_size);
        assert_eq!(aligned, 1024);
        assert!(aligned >= raw_size);
        assert!(is_aligned(aligned));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn page_aligned_sizes_are_also_metal_aligned() {
        // PAGE_SIZE (16 KiB) is a multiple of 256
        assert!(is_aligned(PAGE_SIZE));
        assert_eq!(align_buffer_size(PAGE_SIZE), PAGE_SIZE);
    }
}

// ═════════════════════════════════════════════════════════════════════
// 3. Buffer pool (allocation, reuse, size classes)
// ═════════════════════════════════════════════════════════════════════

mod buffer_pool {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pool_allocate_and_release() {
        let mut pool = BufferPool::new(4096);
        let idx = pool.allocate(100).expect("should allocate");
        assert_eq!(pool.in_use_count(), 1);
        pool.release(idx);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pool_reuses_released_buffer() {
        let mut pool = BufferPool::new(4096);
        let idx1 = pool.allocate(200).unwrap();
        let alloc_after_first = pool.total_allocated();
        pool.release(idx1);

        // Second allocation should reuse the released slot
        let idx2 = pool.allocate(100).unwrap();
        assert_eq!(idx2, idx1, "should reuse the same slot");
        assert_eq!(pool.total_allocated(), alloc_after_first, "no new allocation");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pool_size_classes_escalate() {
        let mut pool = BufferPool::new(1024 * 1024);
        let _a = pool.allocate(100).unwrap(); // → 256
        let _b = pool.allocate(300).unwrap(); // → 512
        let _c = pool.allocate(1000).unwrap(); // → 1024
        assert_eq!(pool.in_use_count(), 3);
        assert_eq!(pool.total_allocated(), 256 + 512 + 1024);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pool_capacity_limit_prevents_overalloc() {
        let mut pool = BufferPool::new(512);
        let _a = pool.allocate(256).unwrap();
        let _b = pool.allocate(256).unwrap();
        let result = pool.allocate(256);
        assert!(result.is_none(), "should fail when capacity exhausted");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pool_fragmentation_tracking() {
        let mut pool = BufferPool::new(4096);
        let a = pool.allocate(256).unwrap();
        let _b = pool.allocate(256).unwrap();
        let _c = pool.allocate(256).unwrap();

        assert_eq!(pool.fragmentation_ratio(), 0.0);

        pool.release(a);
        let frag = pool.fragmentation_ratio();
        assert!(frag > 0.0, "releasing a buffer should increase fragmentation ratio, got {frag}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pool_empty_fragmentation_is_zero() {
        let pool = BufferPool::new(4096);
        assert_eq!(pool.fragmentation_ratio(), 0.0);
    }
}

// ═════════════════════════════════════════════════════════════════════
// 4. Double buffering (ping-pong)
// ═════════════════════════════════════════════════════════════════════

mod double_buffering {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn double_buffer_initial_state() {
        let db = DoubleBuffer::new(1024);
        assert_eq!(db.current, 0);
        assert!(!db.is_in_flight(0));
        assert!(!db.is_in_flight(1));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn double_buffer_swap_alternates() {
        let mut db = DoubleBuffer::new(1024);
        assert_eq!(db.current, 0);
        db.swap();
        assert_eq!(db.current, 1);
        db.complete_flight(0);
        db.swap();
        assert_eq!(db.current, 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn double_buffer_flight_tracking() {
        let mut db = DoubleBuffer::new(512);
        db.swap(); // buffer 0 now in flight
        assert!(db.is_in_flight(0));
        assert!(!db.is_in_flight(1));
        db.complete_flight(0);
        assert!(!db.is_in_flight(0));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn double_buffer_sizes_are_aligned() {
        let db = DoubleBuffer::new(300);
        // 300 → aligned to 512
        assert_eq!(db.buffers[0].len(), 512);
        assert_eq!(db.buffers[1].len(), 512);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn double_buffer_synchronization_semantics() {
        let mut db = DoubleBuffer::new(256);
        // Frame 0: write to buffer 0, then swap
        db.buffers[0][0] = 42;
        db.swap();
        // Frame 1: buffer 1 is now current, buffer 0 is in flight
        assert!(db.is_in_flight(0));
        assert_eq!(db.current_buffer()[0], 0); // buffer 1 untouched
        db.complete_flight(0);
        assert!(!db.is_in_flight(0));
    }
}

// ═════════════════════════════════════════════════════════════════════
// 5. Buffer contents transfer (CPU↔GPU)
// ═════════════════════════════════════════════════════════════════════

mod buffer_contents_transfer {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn cpu_to_gpu_copy_semantics() {
        // Simulate CPU→GPU: write to staging, then "blit" to device buffer
        let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let byte_len = src.len() * F32_BYTES;
        let aligned_len = align_buffer_size(byte_len);
        let mut dst = vec![0u8; aligned_len];
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, byte_len) };
        dst[..byte_len].copy_from_slice(src_bytes);
        // Verify round-trip
        let result: &[f32] =
            unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, src.len()) };
        assert_eq!(result, src.as_slice());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn gpu_to_cpu_readback() {
        // Simulate GPU→CPU readback
        let mut gpu_buf = vec![0u8; 512];
        // "GPU" writes pattern
        for (i, b) in gpu_buf.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let cpu_copy = gpu_buf.clone();
        assert_eq!(cpu_copy, gpu_buf);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn partial_blit_copy() {
        let src = vec![1u8; 1024];
        let mut dst = vec![0u8; 1024];
        let copy_len = 512;
        let src_offset = 256;
        let dst_offset = 0;
        dst[dst_offset..dst_offset + copy_len]
            .copy_from_slice(&src[src_offset..src_offset + copy_len]);
        assert!(dst[..copy_len].iter().all(|&b| b == 1));
        assert!(dst[copy_len..].iter().all(|&b| b == 0));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn synchronize_managed_buffer() {
        // Managed storage requires explicit synchronize after CPU writes
        let desc = BufferDescriptor::new("managed", 256, StorageMode::Managed);
        assert_eq!(desc.storage_mode, StorageMode::Managed);
        // In real Metal: [buffer didModifyRange:] then synchronize
        assert!(is_aligned(desc.aligned_size));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn unified_memory_zero_copy_shared() {
        let mem = MemoryArchitecture::Unified;
        assert!(mem.supports_zero_copy(), "unified memory should support zero-copy buffer sharing");
    }
}

// ═════════════════════════════════════════════════════════════════════
// 6. Buffer offsets (aligned reads, stride-based, multi-tensor)
// ═════════════════════════════════════════════════════════════════════

mod buffer_offsets {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn offset_alignment_at_boundaries() {
        for mult in 0..32 {
            let offset = mult * METAL_BUFFER_ALIGNMENT;
            assert!(is_aligned(offset), "offset {offset} should be aligned");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn stride_based_access_f32() {
        // Accessing f32 elements at stride=256 bytes (64 f32s)
        let stride_bytes = 256;
        let num_rows = 4;
        let total = stride_bytes * num_rows;
        let buf = vec![0u8; total];
        for row in 0..num_rows {
            let offset = row * stride_bytes;
            assert!(is_aligned(offset));
            assert!(offset + stride_bytes <= buf.len());
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn multi_tensor_in_single_buffer() {
        // Pack two tensors into one buffer at aligned offsets
        let tensor_a_bytes = 1000; // → 1024 aligned
        let tensor_b_bytes = 500; // → 512 aligned
        let offset_a = 0usize;
        let offset_b = align_buffer_size(tensor_a_bytes);
        let total = offset_b + align_buffer_size(tensor_b_bytes);

        assert!(is_aligned(offset_a));
        assert!(is_aligned(offset_b));
        assert_eq!(offset_b, 1024);
        assert_eq!(total, 1024 + 512);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn non_aligned_offset_detected() {
        assert!(!is_aligned(1));
        assert!(!is_aligned(128));
        assert!(!is_aligned(255));
        assert!(!is_aligned(257));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn three_tensor_packing_layout() {
        let sizes = [768usize, 1500, 300];
        let mut offset = 0;
        let mut offsets = Vec::new();
        for &s in &sizes {
            assert!(is_aligned(offset));
            offsets.push(offset);
            offset += align_buffer_size(s);
        }
        assert_eq!(offsets, vec![0, 768, 768 + 1536]);
        assert!(is_aligned(offset));
    }
}

// ═════════════════════════════════════════════════════════════════════
// 7. Memory pressure (allocation failure, purging, resource options)
// ═════════════════════════════════════════════════════════════════════

mod memory_pressure {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn allocation_failure_returns_none() {
        let mut pool = BufferPool::new(256);
        let _ = pool.allocate(256).unwrap();
        assert!(pool.allocate(256).is_none());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn purge_releases_unused_buffers() {
        let mut pool = BufferPool::new(2048);
        let a = pool.allocate(256).unwrap();
        let b = pool.allocate(256).unwrap();
        pool.release(a);
        pool.release(b);
        assert_eq!(pool.in_use_count(), 0);
        // After purge, freed buffers can be reclaimed
        let c = pool.allocate(256).unwrap();
        assert_eq!(c, 0); // reuses first slot
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn resource_option_cpu_cache_mode_hint() {
        // On unified memory, default write-combined is preferred for GPU-only buffers
        let mem = MemoryArchitecture::detect();
        if mem == MemoryArchitecture::Unified {
            assert!(mem.supports_zero_copy());
        }
        // Private storage buffers don't need CPU cache coherency
        let desc = BufferDescriptor::new("private_scratch", 4096, StorageMode::Private);
        assert_eq!(desc.storage_mode, StorageMode::Private);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn sequential_allocations_under_pressure() {
        let mut pool = BufferPool::new(1024);
        let mut handles = Vec::new();
        // Fill the pool
        while let Some(h) = pool.allocate(256) {
            handles.push(h);
        }
        assert_eq!(handles.len(), 4); // 1024 / 256 = 4
        // Release one and reallocate
        pool.release(handles[2]);
        let reused = pool.allocate(256).unwrap();
        assert_eq!(reused, handles[2]);
    }
}

// ═════════════════════════════════════════════════════════════════════
// 8. Buffer hazard tracking (RAW, WAR, fencing)
// ═════════════════════════════════════════════════════════════════════

mod buffer_hazard_tracking {
    use super::*;

    /// Tracks pending buffer operations for hazard detection.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum AccessKind {
        Read,
        Write,
    }

    struct HazardTracker {
        last_access: Option<AccessKind>,
        fence_count: u32,
    }

    impl HazardTracker {
        fn new() -> Self {
            Self { last_access: None, fence_count: 0 }
        }

        fn needs_fence(&self, next: AccessKind) -> bool {
            match (self.last_access, next) {
                (Some(AccessKind::Write), AccessKind::Read) => true, // RAW
                (Some(AccessKind::Read), AccessKind::Write) => true, // WAR
                (Some(AccessKind::Write), AccessKind::Write) => true, // WAW
                _ => false,
            }
        }

        fn record(&mut self, kind: AccessKind) {
            if self.needs_fence(kind) {
                self.fence_count += 1;
            }
            self.last_access = Some(kind);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn read_after_write_requires_fence() {
        let tracker = HazardTracker { last_access: Some(AccessKind::Write), fence_count: 0 };
        assert!(tracker.needs_fence(AccessKind::Read));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn write_after_read_requires_fence() {
        let tracker = HazardTracker { last_access: Some(AccessKind::Read), fence_count: 0 };
        assert!(tracker.needs_fence(AccessKind::Write));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn read_after_read_no_fence() {
        let tracker = HazardTracker { last_access: Some(AccessKind::Read), fence_count: 0 };
        assert!(!tracker.needs_fence(AccessKind::Read));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn fence_count_increments_on_hazards() {
        let mut tracker = HazardTracker::new();
        tracker.record(AccessKind::Write);
        assert_eq!(tracker.fence_count, 0); // first access, no prior
        tracker.record(AccessKind::Read); // RAW → fence
        assert_eq!(tracker.fence_count, 1);
        tracker.record(AccessKind::Write); // WAR → fence
        assert_eq!(tracker.fence_count, 2);
    }
}

// ═════════════════════════════════════════════════════════════════════
// 9. Argument buffer encoding
// ═════════════════════════════════════════════════════════════════════

mod argument_buffer_encoding {
    use super::*;

    /// Simulated argument buffer entry (buffer index + offset).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ArgBufferEntry {
        buffer_index: u32,
        offset: usize,
        size: usize,
    }

    fn encode_arg_buffer(entries: &[(u32, usize, usize)]) -> Vec<ArgBufferEntry> {
        entries
            .iter()
            .map(|&(idx, off, sz)| {
                assert!(is_aligned(off), "arg buffer offset {off} must be 256-byte aligned");
                ArgBufferEntry { buffer_index: idx, offset: off, size: sz }
            })
            .collect()
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn encode_single_buffer_argument() {
        let entries = encode_arg_buffer(&[(0, 0, 1024)]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].buffer_index, 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn encode_multiple_buffer_arguments() {
        let entries = encode_arg_buffer(&[(0, 0, 1024), (1, 0, 2048), (2, 256, 512)]);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[2].offset, 256);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn indirect_dispatch_dimensions() {
        // Indirect dispatch: dispatch dimensions stored in a buffer
        let wg = WorkgroupSize::linear(SIMD_GROUP_WIDTH).unwrap();
        let dispatch = DispatchDimensions::for_problem((1024, 1, 1), &wg).unwrap();
        assert_eq!(dispatch.x, 32); // 1024 / 32
        assert_eq!(dispatch.y, 1);
        assert_eq!(dispatch.z, 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn argument_buffer_alignment_enforcement() {
        // All offsets in argument buffer must be 256-byte aligned
        let offsets = [0, 256, 512, 768, 1024];
        for &off in &offsets {
            assert!(is_aligned(off), "offset {off} should be aligned for arg buffer");
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 10. Buffer size limits (max size, 4GB boundary)
// ═════════════════════════════════════════════════════════════════════

mod buffer_size_limits {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn four_gb_limit_check() {
        // Some Metal devices cap individual buffers at 4 GiB
        let limit = METAL_4GB_LIMIT;
        assert_eq!(limit, 4 * 1024 * 1024 * 1024);
        assert!(is_aligned(limit));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn multi_buffer_for_large_tensors() {
        // Split a 6 GiB tensor across two buffers
        let tensor_bytes: usize = 6 * 1024 * 1024 * 1024;
        let num_buffers = tensor_bytes.div_ceil(METAL_4GB_LIMIT);
        assert_eq!(num_buffers, 2);

        let last_buf_size = tensor_bytes - (num_buffers - 1) * METAL_4GB_LIMIT;
        assert!(last_buf_size <= METAL_4GB_LIMIT);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn max_buffer_size_is_aligned() {
        assert!(is_aligned(METAL_4GB_LIMIT));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn aligned_size_never_smaller_than_input() {
        for size in [0, 1, 127, 128, 255, 256, 1023, 1024, 100_000] {
            let aligned = align_buffer_size(size);
            assert!(aligned >= size, "align_buffer_size({size}) = {aligned} should be >= {size}");
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// 11. Buffer label / debug (GPU debugging, capture scope)
// ═════════════════════════════════════════════════════════════════════

mod buffer_label_debug {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_label_assignment() {
        let p = MetalComputePipeline::new("matmul_f32_kernel");
        assert_eq!(p.label, "matmul_f32_kernel");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_label_empty_string() {
        let p = MetalComputePipeline::new("");
        assert_eq!(p.label, "");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_label_unicode() {
        let p = MetalComputePipeline::new("kernel_αβγ_test");
        assert_eq!(p.label, "kernel_αβγ_test");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn buffer_descriptor_labels_for_capture_scope() {
        // In a GPU capture, each buffer should have a descriptive label
        let descriptors = [
            BufferDescriptor::new("input_embeddings", 4096, StorageMode::Shared),
            BufferDescriptor::new("attention_weights", 8192, StorageMode::Private),
            BufferDescriptor::new("output_logits", 2048, StorageMode::Shared),
        ];
        let labels: Vec<&str> = descriptors.iter().map(|d| d.label.as_str()).collect();
        assert_eq!(labels, vec!["input_embeddings", "attention_weights", "output_logits"]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_debug_format_includes_label() {
        let p = MetalComputePipeline::new("my_kernel");
        let debug = format!("{p:?}");
        assert!(
            debug.contains("my_kernel"),
            "Debug output should include pipeline label, got: {debug}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════
// 12. Ring buffer (circular allocation, wrap-around, frame cleanup)
// ═════════════════════════════════════════════════════════════════════

mod ring_buffer_tests {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn ring_buffer_initial_state() {
        let rb = RingBuffer::new(4096);
        assert_eq!(rb.used_bytes(), 0);
        assert_eq!(rb.capacity, metal_align(4096));
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn ring_buffer_allocate_advances_head() {
        let mut rb = RingBuffer::new(4096);
        let off = rb.allocate(100).unwrap();
        assert_eq!(off, 0);
        assert_eq!(rb.used_bytes(), 256); // aligned up
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn ring_buffer_sequential_allocations() {
        let mut rb = RingBuffer::new(4096);
        let o1 = rb.allocate(100).unwrap();
        let o2 = rb.allocate(200).unwrap();
        assert_eq!(o1, 0);
        assert_eq!(o2, 256);
        assert_eq!(rb.used_bytes(), 512);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn ring_buffer_frame_based_cleanup() {
        let mut rb = RingBuffer::new(2048);
        let _ = rb.allocate(256).unwrap();
        rb.end_frame();
        let _ = rb.allocate(256).unwrap();
        rb.end_frame();

        // Retire first frame → tail moves to first frame end
        rb.retire_frame();
        assert_eq!(rb.tail, 256);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn ring_buffer_allocation_failure_when_full() {
        let mut rb = RingBuffer::new(512);
        let _ = rb.allocate(256).unwrap();
        let _ = rb.allocate(256).unwrap();
        assert!(rb.allocate(256).is_none());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn ring_buffer_capacity_is_aligned() {
        let rb = RingBuffer::new(1000);
        assert!(is_aligned(rb.capacity));
        assert_eq!(rb.capacity, 1024);
    }
}

// ═════════════════════════════════════════════════════════════════════
// Cross-cutting: pipeline integration with buffer management
// ═════════════════════════════════════════════════════════════════════

mod pipeline_buffer_integration {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_dispatch_with_aligned_buffers() {
        let p = MetalComputePipeline::new("gemm");
        let m = 128u32;
        let n = 256u32;
        let dispatch = p.dispatch_for_matrix(m, n).unwrap();

        let buf_a = p.aligned_buffer_bytes(m as usize * n as usize, F32_BYTES);
        let buf_b = p.aligned_buffer_bytes(n as usize * n as usize, F32_BYTES);

        assert!(is_aligned(buf_a));
        assert!(is_aligned(buf_b));
        assert!(dispatch.x > 0 && dispatch.y > 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn unified_memory_shared_buffer_preferred() {
        let p = MetalComputePipeline::new("k").with_memory(MemoryArchitecture::Unified);
        assert!(p.memory.supports_zero_copy());
        // Shared storage is preferred on unified memory
        let desc = BufferDescriptor::new("shared_tensor", 2048, StorageMode::Shared);
        assert_eq!(desc.storage_mode, StorageMode::Shared);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn discrete_memory_requires_explicit_transfer() {
        let p = MetalComputePipeline::new("k").with_memory(MemoryArchitecture::Discrete);
        assert!(!p.memory.supports_zero_copy());
    }
}
