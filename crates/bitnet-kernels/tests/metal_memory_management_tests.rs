#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
#![cfg(target_os = "macos")]
#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

/// Helper struct simulating MTLBuffer with alignment tracking
#[derive(Debug, Clone)]
struct GpuBuffer {
    data: Vec<u8>,
    alignment: usize,
}

impl GpuBuffer {
    fn new(size: usize, alignment: usize) -> Self {
        let aligned_size = align_up(size, alignment);
        GpuBuffer { data: vec![0u8; aligned_size], alignment }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

/// Buffer pool allocator simulating Metal buffer pool patterns
#[derive(Debug)]
struct BufferPool {
    buffers: Vec<GpuBuffer>,
    available: Vec<usize>,
    alignment: usize,
    peak_memory: usize,
    current_memory: usize,
}

impl BufferPool {
    fn new(alignment: usize) -> Self {
        BufferPool {
            buffers: Vec::new(),
            available: Vec::new(),
            alignment,
            peak_memory: 0,
            current_memory: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> usize {
        let aligned_size = align_up(size, self.alignment);

        if let Some(idx) = self.available.pop()
            && self.buffers[idx].size() >= aligned_size
        {
            return idx;
        }

        let buffer = GpuBuffer::new(aligned_size, self.alignment);
        self.current_memory += buffer.size();
        if self.current_memory > self.peak_memory {
            self.peak_memory = self.current_memory;
        }

        let idx = self.buffers.len();
        self.buffers.push(buffer);
        idx
    }

    fn release(&mut self, idx: usize) {
        if idx < self.buffers.len() {
            self.available.push(idx);
        }
    }

    fn get(&self, idx: usize) -> Option<&GpuBuffer> {
        if idx < self.buffers.len() { Some(&self.buffers[idx]) } else { None }
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut GpuBuffer> {
        if idx < self.buffers.len() { Some(&mut self.buffers[idx]) } else { None }
    }

    fn total_buffers(&self) -> usize {
        self.buffers.len()
    }

    fn available_count(&self) -> usize {
        self.available.len()
    }

    fn peak_memory_usage(&self) -> usize {
        self.peak_memory
    }

    fn current_memory_usage(&self) -> usize {
        self.current_memory
    }
}

/// Align size upward to the nearest multiple of alignment
#[inline]
fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Align to 4096 byte page boundaries
#[inline]
fn page_align(size: usize) -> usize {
    align_up(size, 4096)
}

/// Calculate row-major stride for M×N matrix
#[inline]
fn row_major_stride(cols: usize, element_size: usize) -> usize {
    align_up(cols * element_size, 64)
}

/// Calculate column-major stride for M×N matrix
#[inline]
fn column_major_stride(rows: usize, element_size: usize) -> usize {
    align_up(rows * element_size, 64)
}

/// Calculate offset within tiled layout (8×8 tiles)
#[inline]
fn tiled_offset(
    tile_row: usize,
    tile_col: usize,
    in_tile_row: usize,
    in_tile_col: usize,
    element_size: usize,
) -> usize {
    let tile_size = 8 * 8 * element_size;
    let tile_stride = tile_col * tile_size;
    let row_stride = tile_row * tile_size * 1024;
    let in_tile_offset = (in_tile_row * 8 + in_tile_col) * element_size;
    row_stride + tile_stride + in_tile_offset
}

/// Calculate memory layout for packed ternary weights
#[inline]
fn packed_weight_offset(bit_index: usize, element_size: usize) -> usize {
    let bits_per_byte = 8;
    let packed_index = bit_index / bits_per_byte;
    align_up(packed_index * element_size, 64)
}

// ============================================================================
// ALIGNMENT TESTS
// ============================================================================

#[cfg(test)]
mod alignment {
    use super::*;

    #[test]
    fn test_align_up_power_of_2() {
        // Test 4096 byte alignment (common for GPU memory)
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(8191, 4096), 8192);
        assert_eq!(align_up(8192, 4096), 8192);
    }

    #[test]
    fn test_align_up_already_aligned() {
        // Already aligned values should remain unchanged
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(512, 256), 512);
        assert_eq!(align_up(1024, 256), 1024);
        assert_eq!(align_up(2048, 256), 2048);
        assert_eq!(align_up(4096, 256), 4096);
    }

    #[test]
    fn test_page_alignment() {
        // Test 4KB page alignment
        assert_eq!(page_align(0), 0);
        assert_eq!(page_align(1), 4096);
        assert_eq!(page_align(4096), 4096);
        assert_eq!(page_align(4097), 8192);
        assert_eq!(page_align(8191), 8192);
        assert_eq!(page_align(16384), 16384);
        assert_eq!(page_align(16385), 20480);
    }

    #[test]
    fn test_cache_line_alignment() {
        // Test 64-byte cache line alignment
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
        assert_eq!(align_up(127, 64), 128);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(256, 64), 256);
    }

    #[test]
    fn test_simd_alignment() {
        // Test 16-byte NEON alignment (f32x4 = 16 bytes)
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(31, 16), 32);
        assert_eq!(align_up(32, 16), 32);
        assert_eq!(align_up(64, 16), 64);
    }
}

// ============================================================================
// BUFFER POOL TESTS
// ============================================================================

#[cfg(test)]
mod buffer_pool {
    use super::*;

    #[test]
    fn test_pool_allocate_and_release() {
        let mut pool = BufferPool::new(64);

        // Allocate buffers
        let idx1 = pool.allocate(1024);
        let _idx2 = pool.allocate(2048);
        let _idx3 = pool.allocate(512);

        assert_eq!(pool.total_buffers(), 3);
        assert_eq!(pool.available_count(), 0);

        // Release buffers
        pool.release(idx1);
        pool.release(_idx2);
        pool.release(_idx3);

        assert_eq!(pool.total_buffers(), 3);
        assert_eq!(pool.available_count(), 3);
    }

    #[test]
    fn test_pool_reuse_released_buffer() {
        let mut pool = BufferPool::new(64);

        // Allocate and release
        let idx1 = pool.allocate(1024);
        pool.release(idx1);

        assert_eq!(pool.available_count(), 1);

        // Allocate again - should reuse the released buffer
        let idx2 = pool.allocate(512);
        assert_eq!(idx2, idx1);
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.total_buffers(), 1);
    }

    #[test]
    fn test_pool_grows_when_needed() {
        let mut pool = BufferPool::new(64);

        let idx1 = pool.allocate(1024);
        pool.release(idx1);

        // Request larger size than available buffer
        let idx2 = pool.allocate(4096);
        assert_ne!(idx2, idx1);
        assert_eq!(pool.total_buffers(), 2);
        assert_eq!(pool.available_count(), 0); // idx1 is still available but not reused
    }

    #[test]
    fn test_pool_respects_alignment() {
        let mut pool = BufferPool::new(256);

        let idx1 = pool.allocate(1000);
        let idx2 = pool.allocate(1000);

        let buf1 = pool.get(idx1).unwrap();
        let buf2 = pool.get(idx2).unwrap();

        // Both buffers should have aligned sizes
        assert_eq!(buf1.size() % 256, 0);
        assert_eq!(buf2.size() % 256, 0);

        // Sizes should be aligned upward
        assert!(buf1.size() >= 1000);
        assert!(buf2.size() >= 1000);
    }
}

// ============================================================================
// MEMORY LAYOUT TESTS
// ============================================================================

#[cfg(test)]
mod memory_layout {
    use super::*;

    #[test]
    fn test_row_major_stride() {
        // Row-major storage for matrix with M rows, N columns
        // Stride should be aligned to 64-byte cache lines

        let stride_100 = row_major_stride(100, 4); // 100 f32s per row
        assert_eq!(stride_100, 448); // (100 * 4 = 400, align up to 448)
        assert_eq!(stride_100 % 64, 0);

        let stride_64 = row_major_stride(64, 4);
        assert_eq!(stride_64, 256); // (64 * 4 = 256, already aligned)
        assert_eq!(stride_64 % 64, 0);

        let stride_65 = row_major_stride(65, 4);
        assert_eq!(stride_65, 320); // (65 * 4 = 260, align up to 320)
        assert_eq!(stride_65 % 64, 0);
    }

    #[test]
    fn test_column_major_stride() {
        // Column-major storage for matrix with M rows, N columns
        // Stride should be aligned to 64-byte cache lines

        let stride_100 = column_major_stride(100, 4); // 100 f32s per column
        assert_eq!(stride_100, 448); // (100 * 4 = 400, align up to 448)
        assert_eq!(stride_100 % 64, 0);

        let stride_64 = column_major_stride(64, 4);
        assert_eq!(stride_64, 256); // (64 * 4 = 256, already aligned)
        assert_eq!(stride_64 % 64, 0);

        let stride_50 = column_major_stride(50, 4);
        assert_eq!(stride_50, 256); // (50 * 4 = 200, align up to 256)
        assert_eq!(stride_50 % 64, 0);
    }

    #[test]
    fn test_tiled_layout_offsets() {
        // 8×8 tiles are common in GPU kernels
        // Test offset calculations for tiled layout

        // First tile (0, 0), in-tile position (0, 0)
        let offset_00 = tiled_offset(0, 0, 0, 0, 4);
        assert_eq!(offset_00, 0);

        // First tile (0, 0), in-tile position (0, 1)
        let offset_01 = tiled_offset(0, 0, 0, 1, 4);
        assert_eq!(offset_01, 4);

        // First tile (0, 0), in-tile position (1, 0)
        let offset_10 = tiled_offset(0, 0, 1, 0, 4);
        assert_eq!(offset_10, 32);

        // Second tile horizontally (0, 1), in-tile (0, 0)
        let tile_size = 8 * 8 * 4;
        let offset_next_tile = tiled_offset(0, 1, 0, 0, 4);
        assert_eq!(offset_next_tile, tile_size);

        // Second tile vertically (1, 0), in-tile (0, 0)
        let offset_next_row = tiled_offset(1, 0, 0, 0, 4);
        assert!(offset_next_row > 0);
    }

    #[test]
    fn test_packed_weight_layout() {
        // Packed ternary weights use multiple bits per weight
        // Layout should be cache-aligned

        let offset_0 = packed_weight_offset(0, 1);
        assert_eq!(offset_0, 0);

        let offset_8 = packed_weight_offset(8, 1);
        assert_eq!(offset_8, 64); // Aligned to 64 bytes

        let offset_16 = packed_weight_offset(16, 1);
        assert_eq!(offset_16, 64); // Both in first cache line

        let offset_512 = packed_weight_offset(512, 1);
        assert_eq!(offset_512, 64); // 512/8 = 64, already aligned

        let offset_520 = packed_weight_offset(520, 1);
        assert_eq!(offset_520, 128); // 520/8 = 65, align up to 128
    }
}

// ============================================================================
// MEMORY PRESSURE TESTS
// ============================================================================

#[cfg(test)]
mod memory_pressure {
    use super::*;

    #[test]
    fn test_peak_memory_tracking() {
        let mut pool = BufferPool::new(256);

        // Initial state
        assert_eq!(pool.peak_memory_usage(), 0);
        assert_eq!(pool.current_memory_usage(), 0);

        // Allocate some buffers
        let idx1 = pool.allocate(1024);
        let _idx2 = pool.allocate(2048);

        let peak_after_two = pool.peak_memory_usage();
        assert!(peak_after_two > 0);

        // Release one
        pool.release(idx1);
        assert_eq!(pool.peak_memory_usage(), peak_after_two);

        // Allocate a larger one
        let _idx3 = pool.allocate(4096);
        let peak_after_three = pool.peak_memory_usage();
        assert!(peak_after_three >= peak_after_two);
    }

    #[test]
    fn test_memory_fragmentation() {
        let mut pool = BufferPool::new(256);

        // Allocate multiple buffers
        let buffers: Vec<_> = (0..10).map(|_| pool.allocate(512)).collect();

        let after_alloc = pool.current_memory_usage();

        // Release alternating buffers
        for (i, &idx) in buffers.iter().enumerate() {
            if i % 2 == 0 {
                pool.release(idx);
            }
        }

        // Available buffers should be reusable without growth
        let _idx = pool.allocate(512);
        let after_realloc = pool.current_memory_usage();

        // Should not have grown significantly
        assert!(after_realloc <= after_alloc + 256);
    }

    #[test]
    fn test_batch_allocation() {
        let mut pool = BufferPool::new(256);

        // Allocate many buffers in batch
        let mut indices = Vec::new();
        for _ in 0..100 {
            indices.push(pool.allocate(256));
        }

        assert_eq!(pool.total_buffers(), 100);
        let peak = pool.peak_memory_usage();

        // Release all
        for idx in indices {
            pool.release(idx);
        }

        // Verify peak was tracked correctly
        assert!(peak > 0);
        assert_eq!(pool.available_count(), 100);

        // Reallocate should mostly reuse
        let mut new_indices = Vec::new();
        for _ in 0..50 {
            new_indices.push(pool.allocate(256));
        }

        assert_eq!(pool.total_buffers(), 100);
    }
}
