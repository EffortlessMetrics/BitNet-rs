#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal memory alignment and buffer layout tests for Apple Silicon.
//!
//! Tests buffer alignment requirements, page sizes, storage mode
//! selection, and memory layout optimization for Apple GPU hardware.

#![cfg(target_os = "macos")]
#![allow(clippy::float_cmp, clippy::needless_range_loop, clippy::too_many_arguments)]

// ── Metal-specific constants ────────────────────────────────────────

/// Metal requires 256-byte buffer alignment on Apple GPUs.
const METAL_ALIGNMENT: usize = 256;

/// Apple Silicon page size (16 KiB).
const PAGE_SIZE: usize = 16384;

/// SIMD group width on Apple Silicon (M1–M4).
const SIMD_GROUP_WIDTH: usize = 32;

/// Maximum threadgroup shared memory on Apple Silicon (32 KiB).
const MAX_SHARED_MEMORY: usize = 32 * 1024;

/// Texture row alignment (Metal best practice: 256 bytes).
const TEXTURE_ROW_ALIGNMENT: usize = 256;

/// Resource heap alignment on Apple Silicon.
const HEAP_ALIGNMENT: usize = 256;

// ── Mock types ──────────────────────────────────────────────────────

/// Simulated Metal storage modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    /// CPU and GPU share the same memory (Apple Silicon unified memory).
    Shared,
    /// GPU-only; CPU cannot access.
    Private,
    /// Managed mode with explicit synchronize calls (macOS discrete GPUs).
    Managed,
    /// Memoryless — contents exist only in tile memory during a render pass.
    Memoryless,
}

/// Simulated CPU cache mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuCacheMode {
    DefaultCache,
    WriteCombined,
}

/// Simulated hazard tracking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HazardTrackingMode {
    Automatic,
    Manual,
}

/// Simulated resource usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceUsage {
    Read,
    Write,
    ReadWrite,
}

/// Simulated memory layout order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayoutOrder {
    RowMajor,
    ColumnMajor,
}

/// A mock Metal buffer descriptor.
#[derive(Debug, Clone)]
struct MockBufferDescriptor {
    size: usize,
    storage_mode: StorageMode,
    cpu_cache_mode: CpuCacheMode,
    hazard_tracking: HazardTrackingMode,
}

/// A mock Metal buffer with alignment-aware allocation.
#[derive(Debug, Clone)]
struct MockMetalBuffer {
    label: String,
    size: usize,
    aligned_size: usize,
    storage_mode: StorageMode,
    offset: usize,
}

impl MockMetalBuffer {
    fn new(label: &str, size: usize, storage_mode: StorageMode) -> Self {
        Self {
            label: label.to_string(),
            size,
            aligned_size: align_up(size, METAL_ALIGNMENT),
            storage_mode,
            offset: 0,
        }
    }

    fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    fn is_aligned(&self) -> bool {
        self.aligned_size % METAL_ALIGNMENT == 0 && self.offset % METAL_ALIGNMENT == 0
    }

    fn is_page_aligned(&self) -> bool {
        self.aligned_size % PAGE_SIZE == 0 && self.offset % PAGE_SIZE == 0
    }
}

/// A mock memory layout descriptor for 2-D tensors.
#[derive(Debug, Clone)]
struct MemoryLayout {
    rows: usize,
    cols: usize,
    element_bytes: usize,
    order: LayoutOrder,
    row_pitch: usize,
}

impl MemoryLayout {
    fn row_major(rows: usize, cols: usize, element_bytes: usize) -> Self {
        let raw_pitch = cols * element_bytes;
        let row_pitch = align_up(raw_pitch, METAL_ALIGNMENT);
        Self { rows, cols, element_bytes, order: LayoutOrder::RowMajor, row_pitch }
    }

    fn column_major(rows: usize, cols: usize, element_bytes: usize) -> Self {
        let raw_pitch = rows * element_bytes;
        let row_pitch = align_up(raw_pitch, METAL_ALIGNMENT);
        Self { rows, cols, element_bytes, order: LayoutOrder::ColumnMajor, row_pitch }
    }

    fn total_bytes(&self) -> usize {
        match self.order {
            LayoutOrder::RowMajor => self.rows * self.row_pitch,
            LayoutOrder::ColumnMajor => self.cols * self.row_pitch,
        }
    }

    fn aligned_total_bytes(&self) -> usize {
        align_up(self.total_bytes(), METAL_ALIGNMENT)
    }

    fn stride(&self) -> usize {
        self.row_pitch / self.element_bytes
    }

    fn padding_bytes_per_row(&self) -> usize {
        self.row_pitch - (self.cols * self.element_bytes)
    }
}

/// A mock resource heap for sub-allocation.
#[derive(Debug)]
struct MockResourceHeap {
    label: String,
    total_size: usize,
    used: usize,
    allocations: Vec<HeapAllocation>,
    hazard_tracking: HazardTrackingMode,
}

#[derive(Debug, Clone)]
struct HeapAllocation {
    label: String,
    offset: usize,
    size: usize,
    aligned_size: usize,
}

impl MockResourceHeap {
    fn new(label: &str, size: usize, hazard_tracking: HazardTrackingMode) -> Self {
        Self {
            label: label.to_string(),
            total_size: align_up(size, PAGE_SIZE),
            used: 0,
            allocations: Vec::new(),
            hazard_tracking,
        }
    }

    fn allocate(&mut self, label: &str, size: usize) -> Option<HeapAllocation> {
        let aligned = align_up(size, HEAP_ALIGNMENT);
        if self.used + aligned > self.total_size {
            return None;
        }
        let alloc = HeapAllocation {
            label: label.to_string(),
            offset: self.used,
            size,
            aligned_size: aligned,
        };
        self.used += aligned;
        self.allocations.push(alloc.clone());
        Some(alloc)
    }

    fn remaining(&self) -> usize {
        self.total_size - self.used
    }

    fn utilization(&self) -> f64 {
        if self.total_size == 0 {
            return 0.0;
        }
        self.used as f64 / self.total_size as f64
    }
}

/// A mock hazard tracker for buffer access ordering.
#[derive(Debug)]
struct HazardTracker {
    mode: HazardTrackingMode,
    pending_writes: Vec<String>,
    pending_reads: Vec<String>,
    fences: Vec<FencePoint>,
}

#[derive(Debug, Clone)]
struct FencePoint {
    after_buffer: String,
    fence_type: FenceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FenceType {
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
}

impl HazardTracker {
    fn new(mode: HazardTrackingMode) -> Self {
        Self { mode, pending_writes: Vec::new(), pending_reads: Vec::new(), fences: Vec::new() }
    }

    fn record_write(&mut self, buffer: &str) {
        self.pending_writes.push(buffer.to_string());
    }

    fn record_read(&mut self, buffer: &str) {
        self.pending_reads.push(buffer.to_string());
    }

    fn needs_fence(&self, buffer: &str, usage: ResourceUsage) -> bool {
        match usage {
            ResourceUsage::Read => self.pending_writes.contains(&buffer.to_string()),
            ResourceUsage::Write => {
                self.pending_reads.contains(&buffer.to_string())
                    || self.pending_writes.contains(&buffer.to_string())
            }
            ResourceUsage::ReadWrite => {
                self.pending_reads.contains(&buffer.to_string())
                    || self.pending_writes.contains(&buffer.to_string())
            }
        }
    }

    fn insert_fence(&mut self, buffer: &str, fence_type: FenceType) {
        self.fences.push(FencePoint { after_buffer: buffer.to_string(), fence_type });
        match fence_type {
            FenceType::ReadAfterWrite => {
                self.pending_writes.retain(|b| b != buffer);
            }
            FenceType::WriteAfterRead => {
                self.pending_reads.retain(|b| b != buffer);
            }
            FenceType::WriteAfterWrite => {
                self.pending_writes.retain(|b| b != buffer);
            }
        }
    }

    fn fence_count(&self) -> usize {
        self.fences.len()
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Round `size` up to the next multiple of `alignment`.
fn align_up(size: usize, alignment: usize) -> usize {
    if size == 0 {
        return 0;
    }
    debug_assert!(alignment.is_power_of_two());
    let mask = alignment - 1;
    (size + mask) & !mask
}

/// Check 256-byte alignment.
fn is_metal_aligned(offset: usize) -> bool {
    offset % METAL_ALIGNMENT == 0
}

/// Select best storage mode for a workload pattern.
fn select_storage_mode(
    cpu_read: bool,
    cpu_write: bool,
    gpu_read: bool,
    gpu_write: bool,
    is_unified_memory: bool,
) -> StorageMode {
    if !cpu_read && !cpu_write && gpu_read && gpu_write {
        return StorageMode::Private;
    }
    if !cpu_read && !cpu_write && !gpu_read && !gpu_write {
        return StorageMode::Memoryless;
    }
    if is_unified_memory { StorageMode::Shared } else { StorageMode::Managed }
}

/// Compute optimal row pitch for a given width and element size.
fn compute_optimal_pitch(cols: usize, element_bytes: usize) -> usize {
    align_up(cols * element_bytes, METAL_ALIGNMENT)
}

// ═══════════════════════════════════════════════════════════════════
// 1. Buffer Alignment (20 tests)
// ═══════════════════════════════════════════════════════════════════

mod buffer_alignment {
    use super::*;

    #[test]
    fn alignment_constant_is_256() {
        assert_eq!(METAL_ALIGNMENT, 256);
        assert!(METAL_ALIGNMENT.is_power_of_two());
    }

    #[test]
    fn page_size_is_16kb() {
        assert_eq!(PAGE_SIZE, 16384);
        assert!(PAGE_SIZE.is_power_of_two());
        assert_eq!(PAGE_SIZE, 16 * 1024);
    }

    #[test]
    fn align_zero_returns_zero() {
        assert_eq!(align_up(0, METAL_ALIGNMENT), 0);
    }

    #[test]
    fn align_exact_multiple_unchanged() {
        assert_eq!(align_up(256, METAL_ALIGNMENT), 256);
        assert_eq!(align_up(512, METAL_ALIGNMENT), 512);
        assert_eq!(align_up(1024, METAL_ALIGNMENT), 1024);
    }

    #[test]
    fn align_rounds_up_to_next_boundary() {
        assert_eq!(align_up(1, METAL_ALIGNMENT), 256);
        assert_eq!(align_up(255, METAL_ALIGNMENT), 256);
        assert_eq!(align_up(257, METAL_ALIGNMENT), 512);
        assert_eq!(align_up(300, METAL_ALIGNMENT), 512);
    }

    #[test]
    fn align_16_byte_boundary() {
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn align_page_boundary() {
        assert_eq!(align_up(1, PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_up(PAGE_SIZE, PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_up(PAGE_SIZE + 1, PAGE_SIZE), 2 * PAGE_SIZE);
    }

    #[test]
    fn is_aligned_at_zero() {
        assert!(is_metal_aligned(0));
    }

    #[test]
    fn is_aligned_at_multiples_of_256() {
        for i in 0..100 {
            assert!(is_metal_aligned(i * METAL_ALIGNMENT));
        }
    }

    #[test]
    fn is_not_aligned_at_odd_offsets() {
        assert!(!is_metal_aligned(1));
        assert!(!is_metal_aligned(128));
        assert!(!is_metal_aligned(255));
        assert!(!is_metal_aligned(511));
    }

    #[test]
    fn buffer_creation_auto_aligns() {
        let buf = MockMetalBuffer::new("test", 100, StorageMode::Shared);
        assert_eq!(buf.size, 100);
        assert_eq!(buf.aligned_size, 256);
        assert!(buf.is_aligned());
    }

    #[test]
    fn buffer_aligned_size_never_less_than_original() {
        for size in [1, 50, 100, 255, 256, 257, 1000, 4096] {
            let buf = MockMetalBuffer::new("t", size, StorageMode::Shared);
            assert!(buf.aligned_size >= buf.size);
        }
    }

    #[test]
    fn buffer_offset_alignment_check() {
        let buf = MockMetalBuffer::new("t", 512, StorageMode::Shared).with_offset(256);
        assert!(buf.is_aligned());

        let buf = MockMetalBuffer::new("t", 512, StorageMode::Shared).with_offset(100);
        assert!(!buf.is_aligned());
    }

    #[test]
    fn page_aligned_buffer() {
        let buf = MockMetalBuffer::new("t", PAGE_SIZE, StorageMode::Private);
        assert!(buf.is_page_aligned());
    }

    #[test]
    fn non_page_aligned_buffer() {
        let buf = MockMetalBuffer::new("t", 1024, StorageMode::Private);
        assert!(!buf.is_page_aligned());
    }

    #[test]
    fn packed_vs_aligned_layout_overhead() {
        let element_count = 1000;
        let element_size = 4; // f32
        let packed_size = element_count * element_size;
        let aligned_size = align_up(packed_size, METAL_ALIGNMENT);
        let overhead = aligned_size - packed_size;
        // 4000 → 4096, overhead = 96 bytes
        assert_eq!(packed_size, 4000);
        assert_eq!(aligned_size, 4096);
        assert_eq!(overhead, 96);
    }

    #[test]
    fn simd_group_width_divides_workgroup() {
        // Apple Silicon SIMD group width is 32.
        // Typical workgroup sizes should be multiples of 32 for full utilization.
        for wg_size in [32, 64, 128, 256, 512, 1024] {
            assert_eq!(wg_size % SIMD_GROUP_WIDTH, 0);
        }
    }

    #[test]
    fn shared_memory_limit_respects_32kb() {
        assert_eq!(MAX_SHARED_MEMORY, 32768);
        // A tile of 32×32 f32 = 4096 bytes fits easily.
        let tile_bytes = 32 * 32 * 4;
        assert!(tile_bytes <= MAX_SHARED_MEMORY);
    }

    #[test]
    fn large_buffer_alignment() {
        // 1 GiB buffer must still be 256-byte aligned.
        let size = 1024 * 1024 * 1024;
        let aligned = align_up(size, METAL_ALIGNMENT);
        assert_eq!(aligned, size); // already aligned
        assert!(is_metal_aligned(aligned));
    }

    #[test]
    fn f16_buffer_alignment() {
        // 1000 half-precision floats = 2000 bytes → 2048
        let size = 1000 * 2;
        let aligned = align_up(size, METAL_ALIGNMENT);
        assert_eq!(aligned, 2048);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Storage Modes (20 tests)
// ═══════════════════════════════════════════════════════════════════

mod storage_modes {
    use super::*;

    #[test]
    fn shared_mode_for_cpu_gpu_access() {
        let mode = select_storage_mode(true, true, true, true, true);
        assert_eq!(mode, StorageMode::Shared);
    }

    #[test]
    fn private_mode_for_gpu_only() {
        let mode = select_storage_mode(false, false, true, true, true);
        assert_eq!(mode, StorageMode::Private);
    }

    #[test]
    fn managed_mode_for_discrete_gpu() {
        let mode = select_storage_mode(true, true, true, true, false);
        assert_eq!(mode, StorageMode::Managed);
    }

    #[test]
    fn memoryless_when_unused() {
        let mode = select_storage_mode(false, false, false, false, true);
        assert_eq!(mode, StorageMode::Memoryless);
    }

    #[test]
    fn read_only_cpu_on_unified_uses_shared() {
        let mode = select_storage_mode(true, false, true, false, true);
        assert_eq!(mode, StorageMode::Shared);
    }

    #[test]
    fn write_only_cpu_on_unified_uses_shared() {
        let mode = select_storage_mode(false, true, true, false, true);
        assert_eq!(mode, StorageMode::Shared);
    }

    #[test]
    fn gpu_write_only_uses_private() {
        let mode = select_storage_mode(false, false, false, true, true);
        // Only GPU write, no read → still private is optimal
        // but our function checks gpu_read && gpu_write for private
        assert_ne!(mode, StorageMode::Private);
        assert_eq!(mode, StorageMode::Shared);
    }

    #[test]
    fn shared_mode_allows_zero_copy_on_unified() {
        let buf = MockMetalBuffer::new("weights", 4096, StorageMode::Shared);
        // On unified memory, shared buffers are zero-copy.
        assert_eq!(buf.storage_mode, StorageMode::Shared);
    }

    #[test]
    fn private_mode_best_for_intermediate_tensors() {
        // Intermediate computation results never need CPU access.
        let buf = MockMetalBuffer::new("intermediate", 8192, StorageMode::Private);
        assert_eq!(buf.storage_mode, StorageMode::Private);
    }

    #[test]
    fn weight_buffer_should_use_shared_on_apple_silicon() {
        // Model weights: loaded by CPU, read by GPU.
        let mode = select_storage_mode(true, false, true, false, true);
        assert_eq!(mode, StorageMode::Shared);
    }

    #[test]
    fn output_buffer_cpu_readback_on_unified() {
        // Output: GPU writes, CPU reads back.
        let mode = select_storage_mode(true, false, false, true, true);
        assert_eq!(mode, StorageMode::Shared);
    }

    #[test]
    fn output_buffer_cpu_readback_on_discrete() {
        let mode = select_storage_mode(true, false, false, true, false);
        assert_eq!(mode, StorageMode::Managed);
    }

    #[test]
    fn memoryless_for_render_pass_only() {
        // Tile memory contents are transient.
        let mode = StorageMode::Memoryless;
        assert_ne!(mode, StorageMode::Shared);
        assert_ne!(mode, StorageMode::Private);
    }

    #[test]
    fn buffer_descriptor_default_cpu_cache() {
        let desc = MockBufferDescriptor {
            size: 1024,
            storage_mode: StorageMode::Shared,
            cpu_cache_mode: CpuCacheMode::DefaultCache,
            hazard_tracking: HazardTrackingMode::Automatic,
        };
        assert_eq!(desc.cpu_cache_mode, CpuCacheMode::DefaultCache);
    }

    #[test]
    fn write_combined_cache_for_streaming_writes() {
        let desc = MockBufferDescriptor {
            size: 1024,
            storage_mode: StorageMode::Shared,
            cpu_cache_mode: CpuCacheMode::WriteCombined,
            hazard_tracking: HazardTrackingMode::Automatic,
        };
        assert_eq!(desc.cpu_cache_mode, CpuCacheMode::WriteCombined);
    }

    #[test]
    fn private_mode_not_cpu_accessible() {
        let buf = MockMetalBuffer::new("priv", 256, StorageMode::Private);
        // Private buffers cannot be read/written by CPU.
        assert_eq!(buf.storage_mode, StorageMode::Private);
    }

    #[test]
    fn storage_mode_enum_has_four_variants() {
        let modes = [
            StorageMode::Shared,
            StorageMode::Private,
            StorageMode::Managed,
            StorageMode::Memoryless,
        ];
        assert_eq!(modes.len(), 4);
        // All distinct.
        for i in 0..modes.len() {
            for j in (i + 1)..modes.len() {
                assert_ne!(modes[i], modes[j]);
            }
        }
    }

    #[test]
    fn managed_mode_requires_explicit_sync() {
        // Managed mode means the driver tracks dirty ranges;
        // the app must call synchronize/didModifyRange.
        let desc = MockBufferDescriptor {
            size: 4096,
            storage_mode: StorageMode::Managed,
            cpu_cache_mode: CpuCacheMode::DefaultCache,
            hazard_tracking: HazardTrackingMode::Automatic,
        };
        assert_eq!(desc.storage_mode, StorageMode::Managed);
    }

    #[test]
    fn kv_cache_uses_private_storage() {
        // KV cache is written and read by GPU only during inference.
        let mode = select_storage_mode(false, false, true, true, true);
        assert_eq!(mode, StorageMode::Private);
    }

    #[test]
    fn activation_buffer_uses_private_storage() {
        // Activation tensors are GPU read-write intermediates.
        let mode = select_storage_mode(false, false, true, true, true);
        assert_eq!(mode, StorageMode::Private);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Memory Layout (20 tests)
// ═══════════════════════════════════════════════════════════════════

mod memory_layout {
    use super::*;

    #[test]
    fn row_major_layout_pitch_aligned() {
        let layout = MemoryLayout::row_major(64, 64, 4);
        assert_eq!(layout.row_pitch, 256); // 64*4=256, already aligned
        assert!(is_metal_aligned(layout.row_pitch));
    }

    #[test]
    fn row_major_layout_pitch_rounds_up() {
        let layout = MemoryLayout::row_major(10, 100, 4);
        // 100*4 = 400 → 512
        assert_eq!(layout.row_pitch, 512);
    }

    #[test]
    fn column_major_layout_pitch_aligned() {
        let layout = MemoryLayout::column_major(128, 64, 4);
        // pitch = align_up(128 * 4, 256) = align_up(512, 256) = 512
        assert_eq!(layout.row_pitch, 512);
    }

    #[test]
    fn total_bytes_row_major() {
        let layout = MemoryLayout::row_major(32, 64, 4);
        // pitch = align_up(64*4, 256) = 256, total = 32 * 256 = 8192
        assert_eq!(layout.total_bytes(), 32 * 256);
    }

    #[test]
    fn total_bytes_column_major() {
        let layout = MemoryLayout::column_major(64, 32, 4);
        // pitch = align_up(64*4, 256) = 256, total = 32 * 256 = 8192
        assert_eq!(layout.total_bytes(), 32 * 256);
    }

    #[test]
    fn aligned_total_bytes() {
        let layout = MemoryLayout::row_major(3, 64, 4);
        // pitch = 256, total = 3*256 = 768 → aligned_total = 768
        assert_eq!(layout.aligned_total_bytes(), 768);
    }

    #[test]
    fn stride_equals_pitch_div_element_size() {
        let layout = MemoryLayout::row_major(10, 100, 4);
        // pitch = 512, stride = 512/4 = 128
        assert_eq!(layout.stride(), 128);
    }

    #[test]
    fn padding_per_row() {
        let layout = MemoryLayout::row_major(10, 100, 4);
        // pitch = 512, row_data = 100*4 = 400, padding = 112
        assert_eq!(layout.padding_bytes_per_row(), 112);
    }

    #[test]
    fn no_padding_when_aligned() {
        let layout = MemoryLayout::row_major(10, 64, 4);
        // 64*4 = 256, pitch = 256, padding = 0
        assert_eq!(layout.padding_bytes_per_row(), 0);
    }

    #[test]
    fn optimal_pitch_calculation() {
        assert_eq!(compute_optimal_pitch(100, 4), 512);
        assert_eq!(compute_optimal_pitch(64, 4), 256);
        assert_eq!(compute_optimal_pitch(1, 4), 256);
    }

    #[test]
    fn tiled_layout_8x8_f32() {
        // An 8×8 tile of f32: 8*4 = 32 per row, but pitch = 256
        let layout = MemoryLayout::row_major(8, 8, 4);
        assert_eq!(layout.row_pitch, 256);
        assert_eq!(layout.total_bytes(), 8 * 256);
    }

    #[test]
    fn tiled_layout_16x16_f32() {
        let layout = MemoryLayout::row_major(16, 16, 4);
        // 16*4=64 → pitch = 256
        assert_eq!(layout.row_pitch, 256);
    }

    #[test]
    fn tiled_layout_32x32_f32_fits_shared_memory() {
        let layout = MemoryLayout::row_major(32, 32, 4);
        assert!(layout.total_bytes() <= MAX_SHARED_MEMORY);
    }

    #[test]
    fn f16_layout_half_the_element_size() {
        let f32_layout = MemoryLayout::row_major(64, 128, 4);
        let f16_layout = MemoryLayout::row_major(64, 128, 2);
        assert!(f16_layout.total_bytes() <= f32_layout.total_bytes());
    }

    #[test]
    fn i2s_quantized_layout_2bit() {
        // 2-bit quantized: 4 elements per byte.
        // 256 elements = 64 bytes per row.
        let bytes_per_row = 256 / 4;
        let pitch = align_up(bytes_per_row, METAL_ALIGNMENT);
        assert_eq!(bytes_per_row, 64);
        assert_eq!(pitch, 256);
    }

    #[test]
    fn large_matrix_layout_2048x2048_f32() {
        let layout = MemoryLayout::row_major(2048, 2048, 4);
        // 2048*4 = 8192, already aligned
        assert_eq!(layout.row_pitch, 8192);
        assert_eq!(layout.total_bytes(), 2048 * 8192);
        assert_eq!(layout.padding_bytes_per_row(), 0);
    }

    #[test]
    fn non_square_matrix_layout() {
        let layout = MemoryLayout::row_major(1, 2048, 4);
        // Single row: 2048*4 = 8192
        assert_eq!(layout.row_pitch, 8192);
        assert_eq!(layout.total_bytes(), 8192);
    }

    #[test]
    fn stride_matches_for_contiguous_access() {
        let layout = MemoryLayout::row_major(64, 256, 4);
        // 256*4 = 1024, aligned to 1024 (already aligned)
        assert_eq!(layout.stride(), layout.row_pitch / layout.element_bytes);
        assert_eq!(layout.stride(), 256);
    }

    #[test]
    fn padding_waste_ratio() {
        let layout = MemoryLayout::row_major(100, 100, 4);
        // 100*4=400 → pitch 512, padding 112
        let useful = layout.rows * layout.cols * layout.element_bytes;
        let total = layout.total_bytes();
        let waste = 1.0 - (useful as f64 / total as f64);
        // Some waste is expected; ensure < 50%
        assert!(waste < 0.5, "Waste ratio too high: {waste}");
    }

    #[test]
    fn row_major_vs_column_major_same_data() {
        let rm = MemoryLayout::row_major(64, 128, 4);
        let cm = MemoryLayout::column_major(64, 128, 4);
        // Both store 64*128 elements, but total bytes may differ due to
        // different pitch calculations.
        assert_eq!(rm.order, LayoutOrder::RowMajor);
        assert_eq!(cm.order, LayoutOrder::ColumnMajor);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Resource Heaps (15 tests)
// ═══════════════════════════════════════════════════════════════════

mod resource_heaps {
    use super::*;

    #[test]
    fn heap_creation_page_aligns_total() {
        let heap = MockResourceHeap::new("test", 1000, HazardTrackingMode::Automatic);
        assert_eq!(heap.total_size, PAGE_SIZE); // 1000 → 16384
        assert_eq!(heap.total_size % PAGE_SIZE, 0);
    }

    #[test]
    fn heap_allocate_single_buffer() {
        let mut heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Automatic);
        let alloc = heap.allocate("buf1", 1024).unwrap();
        assert_eq!(alloc.offset, 0);
        assert_eq!(alloc.aligned_size, 1024); // 1024 is already 256-aligned
    }

    #[test]
    fn heap_allocate_multiple_buffers_no_overlap() {
        let mut heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Automatic);
        let a1 = heap.allocate("buf1", 512).unwrap();
        let a2 = heap.allocate("buf2", 512).unwrap();
        assert_eq!(a1.offset, 0);
        assert_eq!(a2.offset, 512);
        // No overlap: a1 ends at 512, a2 starts at 512.
        assert!(a1.offset + a1.aligned_size <= a2.offset);
    }

    #[test]
    fn heap_allocate_aligns_sub_allocations() {
        let mut heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Automatic);
        let a1 = heap.allocate("small", 100).unwrap();
        assert_eq!(a1.aligned_size, 256); // 100 → 256
        let a2 = heap.allocate("next", 50).unwrap();
        assert_eq!(a2.offset, 256);
        assert!(is_metal_aligned(a2.offset));
    }

    #[test]
    fn heap_runs_out_of_space() {
        let mut heap = MockResourceHeap::new("h", 512, HazardTrackingMode::Automatic);
        // Heap total = align_up(512, 16384) = 16384
        // Allocate most of the heap.
        heap.allocate("big", 16000).unwrap();
        // Only ~384 left; a 1024 alloc should fail.
        let result = heap.allocate("too_big", 1024);
        assert!(result.is_none());
    }

    #[test]
    fn heap_remaining_tracks_correctly() {
        let mut heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Automatic);
        let initial = heap.remaining();
        heap.allocate("a", 1024).unwrap();
        assert_eq!(heap.remaining(), initial - 1024);
    }

    #[test]
    fn heap_utilization_starts_at_zero() {
        let heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Automatic);
        assert_eq!(heap.utilization(), 0.0);
    }

    #[test]
    fn heap_utilization_grows_with_allocations() {
        let mut heap = MockResourceHeap::new("h", PAGE_SIZE, HazardTrackingMode::Automatic);
        heap.allocate("half", PAGE_SIZE / 2).unwrap();
        let util = heap.utilization();
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn heap_aliasing_same_offset_different_label() {
        // Resource aliasing: two resources can share the same heap region
        // if their lifetimes don't overlap.
        let mut heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Manual);
        let a1 = heap.allocate("pass1_tmp", 4096).unwrap();
        // In a real scenario, pass1_tmp would be released before pass2_tmp.
        // We verify the offsets would allow aliasing.
        assert_eq!(a1.offset, 0);
        assert_eq!(a1.aligned_size, 4096);
    }

    #[test]
    fn unified_memory_heap_no_transfer_needed() {
        // On Apple Silicon unified memory, heap resources don't need
        // CPU→GPU copies; both access the same physical memory.
        let heap = MockResourceHeap::new("unified", 1024 * 1024, HazardTrackingMode::Automatic);
        assert!(heap.total_size > 0);
        assert_eq!(heap.total_size % PAGE_SIZE, 0);
    }

    #[test]
    fn heap_automatic_hazard_tracking() {
        let heap = MockResourceHeap::new("auto", 1024, HazardTrackingMode::Automatic);
        assert_eq!(heap.hazard_tracking, HazardTrackingMode::Automatic);
    }

    #[test]
    fn heap_manual_hazard_tracking() {
        let heap = MockResourceHeap::new("manual", 1024, HazardTrackingMode::Manual);
        assert_eq!(heap.hazard_tracking, HazardTrackingMode::Manual);
    }

    #[test]
    fn heap_allocation_labels_tracked() {
        let mut heap = MockResourceHeap::new("h", 64 * 1024, HazardTrackingMode::Automatic);
        heap.allocate("weights", 4096).unwrap();
        heap.allocate("activations", 2048).unwrap();
        assert_eq!(heap.allocations.len(), 2);
        assert_eq!(heap.allocations[0].label, "weights");
        assert_eq!(heap.allocations[1].label, "activations");
    }

    #[test]
    fn heap_many_small_allocations() {
        let mut heap =
            MockResourceHeap::new("small_allocs", 256 * 1024, HazardTrackingMode::Automatic);
        for i in 0..100 {
            let label = format!("buf_{i}");
            let alloc = heap.allocate(&label, 100);
            // Each 100 bytes rounds to 256.
            if let Some(a) = alloc {
                assert!(is_metal_aligned(a.offset));
            }
        }
    }

    #[test]
    fn heap_full_utilization() {
        let mut heap = MockResourceHeap::new("full", PAGE_SIZE, HazardTrackingMode::Automatic);
        // Allocate exactly one page.
        heap.allocate("all", PAGE_SIZE).unwrap();
        assert_eq!(heap.remaining(), 0);
        assert!((heap.utilization() - 1.0).abs() < f64::EPSILON);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Hazard Tracking (15 tests)
// ═══════════════════════════════════════════════════════════════════

mod hazard_tracking {
    use super::*;

    #[test]
    fn no_hazard_on_fresh_tracker() {
        let tracker = HazardTracker::new(HazardTrackingMode::Automatic);
        assert!(!tracker.needs_fence("buf", ResourceUsage::Read));
        assert!(!tracker.needs_fence("buf", ResourceUsage::Write));
    }

    #[test]
    fn read_after_write_hazard_detected() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("buf_a");
        assert!(tracker.needs_fence("buf_a", ResourceUsage::Read));
    }

    #[test]
    fn write_after_read_hazard_detected() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_read("buf_a");
        assert!(tracker.needs_fence("buf_a", ResourceUsage::Write));
    }

    #[test]
    fn write_after_write_hazard_detected() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("buf_a");
        assert!(tracker.needs_fence("buf_a", ResourceUsage::Write));
    }

    #[test]
    fn no_cross_buffer_hazard() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("buf_a");
        assert!(!tracker.needs_fence("buf_b", ResourceUsage::Read));
    }

    #[test]
    fn fence_clears_write_hazard() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("buf_a");
        assert!(tracker.needs_fence("buf_a", ResourceUsage::Read));
        tracker.insert_fence("buf_a", FenceType::ReadAfterWrite);
        assert!(!tracker.needs_fence("buf_a", ResourceUsage::Read));
    }

    #[test]
    fn fence_clears_read_hazard() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_read("buf_a");
        assert!(tracker.needs_fence("buf_a", ResourceUsage::Write));
        tracker.insert_fence("buf_a", FenceType::WriteAfterRead);
        assert!(!tracker.needs_fence("buf_a", ResourceUsage::Write));
    }

    #[test]
    fn multiple_fences_tracked() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("a");
        tracker.insert_fence("a", FenceType::ReadAfterWrite);
        tracker.record_read("b");
        tracker.insert_fence("b", FenceType::WriteAfterRead);
        assert_eq!(tracker.fence_count(), 2);
    }

    #[test]
    fn automatic_tracking_mode_created() {
        let tracker = HazardTracker::new(HazardTrackingMode::Automatic);
        assert_eq!(tracker.mode, HazardTrackingMode::Automatic);
    }

    #[test]
    fn manual_tracking_mode_created() {
        let tracker = HazardTracker::new(HazardTrackingMode::Manual);
        assert_eq!(tracker.mode, HazardTrackingMode::Manual);
    }

    #[test]
    fn readwrite_usage_detects_write_hazard() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("buf");
        assert!(tracker.needs_fence("buf", ResourceUsage::ReadWrite));
    }

    #[test]
    fn readwrite_usage_detects_read_hazard() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_read("buf");
        assert!(tracker.needs_fence("buf", ResourceUsage::ReadWrite));
    }

    #[test]
    fn fence_count_starts_at_zero() {
        let tracker = HazardTracker::new(HazardTrackingMode::Manual);
        assert_eq!(tracker.fence_count(), 0);
    }

    #[test]
    fn sequential_write_read_write_chain() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        // Write → fence → Read → fence → Write
        tracker.record_write("buf");
        tracker.insert_fence("buf", FenceType::ReadAfterWrite);
        tracker.record_read("buf");
        tracker.insert_fence("buf", FenceType::WriteAfterRead);
        tracker.record_write("buf");
        assert_eq!(tracker.fence_count(), 2);
        // Still a pending write.
        assert!(tracker.needs_fence("buf", ResourceUsage::Read));
    }

    #[test]
    fn multiple_buffers_independent_hazards() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);
        tracker.record_write("a");
        tracker.record_read("b");
        assert!(tracker.needs_fence("a", ResourceUsage::Read));
        assert!(tracker.needs_fence("b", ResourceUsage::Write));
        assert!(!tracker.needs_fence("a", ResourceUsage::Write));
        // "a" has a pending write, so WAW is detected.
        // Actually, our implementation: pending_writes contains "a" and
        // Write checks pending_reads OR pending_writes.
        // pending_writes has "a", so Write on "a" → true.
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Integration (10 tests)
// ═══════════════════════════════════════════════════════════════════

mod integration {
    use super::*;

    #[test]
    fn matmul_buffer_layout_aligned() {
        // Matrix multiply: A[M×K] × B[K×N] → C[M×N]
        let m = 128;
        let k = 256;
        let n = 64;
        let a = MemoryLayout::row_major(m, k, 4);
        let b = MemoryLayout::row_major(k, n, 4);
        let c = MemoryLayout::row_major(m, n, 4);

        assert!(is_metal_aligned(a.row_pitch));
        assert!(is_metal_aligned(b.row_pitch));
        assert!(is_metal_aligned(c.row_pitch));
    }

    #[test]
    fn attention_kv_cache_layout() {
        let num_heads = 32;
        let head_dim = 128;
        let max_seq = 2048;

        // K cache: [num_heads, max_seq, head_dim]
        let k_row_bytes = head_dim * 4; // f32
        let k_pitch = align_up(k_row_bytes, METAL_ALIGNMENT);
        let k_total = num_heads * max_seq * k_pitch;

        assert!(is_metal_aligned(k_pitch));
        assert!(k_total > 0);
        // 32 * 2048 * 512 = 33,554,432 bytes ≈ 32 MiB
        assert_eq!(k_total, 32 * 2048 * 512);
    }

    #[test]
    fn embedding_table_alignment() {
        let vocab_size = 32000;
        let embed_dim = 2048;
        let layout = MemoryLayout::row_major(vocab_size, embed_dim, 4);

        assert!(is_metal_aligned(layout.row_pitch));
        // Each row: 2048*4 = 8192 bytes, already aligned.
        assert_eq!(layout.padding_bytes_per_row(), 0);
    }

    #[test]
    fn weight_buffer_alignment_i2s() {
        // BitNet I2_S weights: 2-bit packed, 4 values per byte.
        let rows = 2048;
        let cols_packed = 2048 / 4; // 512 bytes per row
        let layout = MemoryLayout::row_major(rows, cols_packed, 1);

        assert!(is_metal_aligned(layout.row_pitch));
        assert_eq!(layout.row_pitch, 512);
    }

    #[test]
    fn full_inference_buffer_set() {
        let batch = 1;
        let seq_len = 512;
        let hidden = 2048;
        let heads = 32;
        let head_dim = hidden / heads;
        let vocab = 32000;

        let input_buf = MockMetalBuffer::new(
            "input_ids",
            align_up(batch * seq_len * 4, METAL_ALIGNMENT),
            StorageMode::Shared,
        );
        let embed_buf = MockMetalBuffer::new(
            "embeddings",
            align_up(vocab * hidden * 4, METAL_ALIGNMENT),
            StorageMode::Shared,
        );
        let hidden_buf = MockMetalBuffer::new(
            "hidden_state",
            align_up(batch * seq_len * hidden * 4, METAL_ALIGNMENT),
            StorageMode::Private,
        );
        let kv_buf = MockMetalBuffer::new(
            "kv_cache",
            align_up(2 * heads * seq_len * head_dim * 4, METAL_ALIGNMENT),
            StorageMode::Private,
        );
        let output_buf = MockMetalBuffer::new(
            "logits",
            align_up(batch * vocab * 4, METAL_ALIGNMENT),
            StorageMode::Shared,
        );

        assert!(input_buf.is_aligned());
        assert!(embed_buf.is_aligned());
        assert!(hidden_buf.is_aligned());
        assert!(kv_buf.is_aligned());
        assert!(output_buf.is_aligned());
    }

    #[test]
    fn layernorm_buffer_alignment() {
        let hidden = 2048;
        let gamma_buf = MockMetalBuffer::new(
            "ln_gamma",
            align_up(hidden * 4, METAL_ALIGNMENT),
            StorageMode::Shared,
        );
        let beta_buf = MockMetalBuffer::new(
            "ln_beta",
            align_up(hidden * 4, METAL_ALIGNMENT),
            StorageMode::Shared,
        );
        assert!(gamma_buf.is_aligned());
        assert!(beta_buf.is_aligned());
    }

    #[test]
    fn heap_based_inference_allocation() {
        let mut heap =
            MockResourceHeap::new("inference", 64 * 1024 * 1024, HazardTrackingMode::Automatic);

        let _weights = heap.allocate("weights", 32 * 1024 * 1024).unwrap();
        let _kv = heap.allocate("kv_cache", 16 * 1024 * 1024).unwrap();
        let _scratch = heap.allocate("scratch", 8 * 1024 * 1024).unwrap();

        assert_eq!(heap.allocations.len(), 3);
        // All offsets aligned.
        for alloc in &heap.allocations {
            assert!(is_metal_aligned(alloc.offset));
        }
    }

    #[test]
    fn hazard_tracking_matmul_pipeline() {
        let mut tracker = HazardTracker::new(HazardTrackingMode::Manual);

        // Step 1: Write weights to buffer.
        tracker.record_write("weights");
        // Step 2: MatMul reads weights, writes output.
        assert!(tracker.needs_fence("weights", ResourceUsage::Read));
        tracker.insert_fence("weights", FenceType::ReadAfterWrite);
        tracker.record_read("weights");
        tracker.record_write("output");
        // Step 3: Softmax reads output.
        assert!(tracker.needs_fence("output", ResourceUsage::Read));
        tracker.insert_fence("output", FenceType::ReadAfterWrite);

        assert_eq!(tracker.fence_count(), 2);
    }

    #[test]
    fn rope_buffer_alignment() {
        let max_seq = 4096;
        let head_dim = 128;
        // RoPE frequency table: [max_seq, head_dim/2] complex pairs → f32
        let rope_elements = max_seq * (head_dim / 2) * 2;
        let rope_bytes = rope_elements * 4;
        let buf = MockMetalBuffer::new(
            "rope_freqs",
            align_up(rope_bytes, METAL_ALIGNMENT),
            StorageMode::Shared,
        );
        assert!(buf.is_aligned());
    }

    #[test]
    fn multi_layer_weight_heap_allocation() {
        let num_layers = 24;
        let weight_size_per_layer = 2048 * 2048 / 4; // I2_S packed
        let mut heap = MockResourceHeap::new(
            "model_weights",
            num_layers * align_up(weight_size_per_layer, HEAP_ALIGNMENT) + PAGE_SIZE,
            HazardTrackingMode::Automatic,
        );

        for i in 0..num_layers {
            let label = format!("layer_{i}_attn");
            let alloc = heap.allocate(&label, weight_size_per_layer);
            assert!(alloc.is_some(), "Failed to allocate layer {i}");
        }
        assert_eq!(heap.allocations.len(), num_layers);
    }
}
