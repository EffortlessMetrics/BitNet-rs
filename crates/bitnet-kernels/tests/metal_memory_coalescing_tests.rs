#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal GPU memory coalescing validation tests.
//!
//! Pure-Rust tests that model Apple Metal GPU memory access patterns,
//! buffer alignment, threadgroup layouts, and coalescing behaviour.
//! These validate the mathematical correctness of dispatch and layout
//! decisions without requiring Metal hardware.

#![cfg(feature = "cpu")]

// ── Metal modelling structs ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct MetalBufferLayout {
    size: usize,
    alignment: usize,
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ThreadgroupConfig {
    width: u32,
    height: u32,
    depth: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DispatchConfig {
    grid: [u32; 3],
    threadgroup: [u32; 3],
}

#[derive(Debug, Clone, Copy)]
struct CoalescingMetrics {
    efficiency: f32,
    bank_conflicts: u32,
    transactions: u32,
}

// ── Metal constants ─────────────────────────────────────────────────

const METAL_MIN_BUFFER_ALIGNMENT: usize = 16;
const METAL_OPTIMAL_ALIGNMENT: usize = 256;
const METAL_PAGE_SIZE: usize = 4096;
const METAL_SIMDGROUP_SIZE: u32 = 32;
const METAL_MAX_THREADGROUP_SIZE: u32 = 1024;
const METAL_THREADGROUP_MEMORY_BANKS: u32 = 32;
const METAL_CACHE_LINE_BYTES: usize = 128;

// ── Helper functions ────────────────────────────────────────────────

fn align_up(value: usize, alignment: usize) -> usize {
    assert!(alignment.is_power_of_two(), "alignment must be power of 2");
    (value + alignment - 1) & !(alignment - 1)
}

fn is_aligned(value: usize, alignment: usize) -> bool {
    value % alignment == 0
}

fn compute_buffer_layout(
    element_count: usize,
    element_size: usize,
    alignment: usize,
) -> MetalBufferLayout {
    let raw_size = element_count * element_size;
    let aligned_size = align_up(raw_size, alignment);
    MetalBufferLayout { size: aligned_size, alignment, offset: 0 }
}

fn compute_buffer_layout_with_offset(
    element_count: usize,
    element_size: usize,
    alignment: usize,
    offset: usize,
) -> MetalBufferLayout {
    let raw_size = element_count * element_size;
    let aligned_size = align_up(raw_size, alignment);
    let aligned_offset = align_up(offset, alignment);
    MetalBufferLayout { size: aligned_size, alignment, offset: aligned_offset }
}

fn round_to_page(size: usize) -> usize {
    align_up(size, METAL_PAGE_SIZE)
}

fn coalesced_stride_f32(width: usize) -> usize {
    align_up(width * 4, METAL_CACHE_LINE_BYTES) / 4
}

fn compute_threadgroup_for_coalescing(tensor_width: u32, tensor_height: u32) -> ThreadgroupConfig {
    // Prefer width-first for row-major coalescing
    let w = METAL_SIMDGROUP_SIZE.min(tensor_width);
    let h = (METAL_MAX_THREADGROUP_SIZE / w).min(tensor_height).max(1);
    ThreadgroupConfig { width: w, height: h, depth: 1 }
}

fn compute_dispatch(tensor_dims: [u32; 3], tg: &ThreadgroupConfig) -> DispatchConfig {
    let grid = [
        (tensor_dims[0] + tg.width - 1) / tg.width,
        (tensor_dims[1] + tg.height - 1) / tg.height,
        (tensor_dims[2] + tg.depth - 1) / tg.depth,
    ];
    DispatchConfig { grid, threadgroup: [tg.width, tg.height, tg.depth] }
}

fn coalescing_efficiency_row_major(width: usize, threads_per_row: usize) -> f32 {
    // Each SIMD group accesses consecutive elements; efficiency is
    // fraction of a cache-line actually used per transaction.
    let elements_per_line = METAL_CACHE_LINE_BYTES / 4; // f32
    let full_lines = threads_per_row / elements_per_line;
    let remainder = threads_per_row % elements_per_line;
    let total_transactions = full_lines + if remainder > 0 { 1 } else { 0 };
    let useful = (full_lines * elements_per_line + remainder) as f32;
    let transferred = (total_transactions * elements_per_line) as f32;
    if transferred == 0.0 {
        return 0.0;
    }
    (useful / transferred).min(1.0)
}

fn coalescing_efficiency_column_major(height: usize, stride: usize, threads_in_simd: usize) -> f32 {
    // Column-major: consecutive threads hit different rows → each
    // thread touches a different cache line → worst case.
    let distinct_lines: usize = threads_in_simd
        .min(height)
        .min((stride * 4 + METAL_CACHE_LINE_BYTES - 1) / METAL_CACHE_LINE_BYTES);
    let useful_bytes = threads_in_simd * 4;
    let transferred_bytes = distinct_lines * METAL_CACHE_LINE_BYTES;
    if transferred_bytes == 0 {
        return 0.0;
    }
    (useful_bytes as f32 / transferred_bytes as f32).min(1.0)
}

fn bank_conflict_count(stride_elements: u32, threads: u32) -> u32 {
    // Each bank is 4 bytes wide; 32 banks.
    // Conflicts occur when multiple threads hit the same bank.
    let mut bank_hits = [0u32; 32];
    for t in 0..threads {
        let addr = t * stride_elements;
        let bank = addr % METAL_THREADGROUP_MEMORY_BANKS;
        bank_hits[bank as usize] += 1;
    }
    bank_hits.iter().map(|&h| if h > 1 { h - 1 } else { 0 }).sum()
}

fn threadgroup_memory_padded_stride(width: u32, element_bytes: u32) -> u32 {
    let raw_stride = width * element_bytes;
    let bank_width = 4u32; // 4 bytes per bank
    let banks = METAL_THREADGROUP_MEMORY_BANKS;
    let stride_in_banks = raw_stride / bank_width;
    if stride_in_banks % banks == 0 {
        // Add 1 bank of padding to break conflicts
        raw_stride + bank_width
    } else {
        raw_stride
    }
}

fn analyse_access_pattern(offsets: &[usize]) -> CoalescingMetrics {
    if offsets.is_empty() {
        return CoalescingMetrics { efficiency: 0.0, bank_conflicts: 0, transactions: 0 };
    }
    // Convert element offsets to byte offsets (f32 = 4 bytes)
    let byte_offsets: Vec<usize> = offsets.iter().map(|o| o * 4).collect();
    let mut lines: Vec<usize> = byte_offsets.iter().map(|o| o / METAL_CACHE_LINE_BYTES).collect();
    lines.sort();
    lines.dedup();
    let transactions = lines.len() as u32;
    let useful = offsets.len() as f32 * 4.0;
    let transferred = transactions as f32 * METAL_CACHE_LINE_BYTES as f32;
    let efficiency = if transferred > 0.0 { (useful / transferred).min(1.0) } else { 0.0 };

    let mut bank_hits = [0u32; 32];
    for &bo in &byte_offsets {
        let bank = (bo / 4) as u32 % METAL_THREADGROUP_MEMORY_BANKS;
        bank_hits[bank as usize] += 1;
    }
    let bank_conflicts = bank_hits.iter().map(|&h| if h > 1 { h - 1 } else { 0 }).sum();

    CoalescingMetrics { efficiency, bank_conflicts, transactions }
}

fn reduction_coalescing_efficiency(input_len: usize, threadgroup_size: usize) -> f32 {
    // First pass: each thread loads one element → fully coalesced if
    // contiguous.  Subsequent tree-reduction passes touch threadgroup
    // memory only.
    let elements_per_line = METAL_CACHE_LINE_BYTES / 4;
    let lines_needed = (threadgroup_size + elements_per_line - 1) / elements_per_line;
    let useful = threadgroup_size.min(input_len) as f32 * 4.0;
    let transferred = lines_needed as f32 * METAL_CACHE_LINE_BYTES as f32;
    if transferred == 0.0 { 0.0 } else { (useful / transferred).min(1.0) }
}

fn batch_matmul_transactions(
    m: u32,
    n: u32,
    k: u32,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    batch: u32,
) -> u32 {
    let tiles_m = (m + tile_m - 1) / tile_m;
    let tiles_n = (n + tile_n - 1) / tile_n;
    let tiles_k = (k + tile_k - 1) / tile_k;
    // Per tile: load tile_m*tile_k from A + tile_k*tile_n from B
    let loads_per_tile = tile_m * tile_k + tile_k * tile_n;
    let elements_per_line = (METAL_CACHE_LINE_BYTES / 4) as u32;
    let transactions_per_tile = (loads_per_tile + elements_per_line - 1) / elements_per_line;
    batch * tiles_m * tiles_n * tiles_k * transactions_per_tile
}

// ═══════════════════════════════════════════════════════════════════
// A. Buffer Alignment Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod buffer_alignment {
    use super::*;

    #[test]
    fn test_16b_alignment_smallest() {
        let layout = compute_buffer_layout(1, 4, METAL_MIN_BUFFER_ALIGNMENT);
        assert!(is_aligned(layout.size, METAL_MIN_BUFFER_ALIGNMENT));
        assert_eq!(layout.size, 16);
    }

    #[test]
    fn test_16b_alignment_exact_boundary() {
        let layout = compute_buffer_layout(4, 4, METAL_MIN_BUFFER_ALIGNMENT);
        assert_eq!(layout.size, 16);
    }

    #[test]
    fn test_16b_alignment_just_over() {
        let layout = compute_buffer_layout(5, 4, METAL_MIN_BUFFER_ALIGNMENT);
        assert_eq!(layout.size, 32);
    }

    #[test]
    fn test_256b_alignment_small_buffer() {
        let layout = compute_buffer_layout(1, 4, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 256);
        assert!(is_aligned(layout.size, METAL_OPTIMAL_ALIGNMENT));
    }

    #[test]
    fn test_256b_alignment_exact() {
        let layout = compute_buffer_layout(64, 4, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 256);
    }

    #[test]
    fn test_256b_alignment_overflow() {
        let layout = compute_buffer_layout(65, 4, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 512);
    }

    #[test]
    fn test_page_alignment_small() {
        let layout = compute_buffer_layout(1, 4, METAL_PAGE_SIZE);
        assert_eq!(layout.size, METAL_PAGE_SIZE);
    }

    #[test]
    fn test_page_alignment_exact() {
        let layout = compute_buffer_layout(1024, 4, METAL_PAGE_SIZE);
        assert_eq!(layout.size, METAL_PAGE_SIZE);
    }

    #[test]
    fn test_page_alignment_overflow() {
        let layout = compute_buffer_layout(1025, 4, METAL_PAGE_SIZE);
        assert_eq!(layout.size, 2 * METAL_PAGE_SIZE);
    }

    #[test]
    fn test_stride_for_coalesced_access_small() {
        let stride = coalesced_stride_f32(8);
        assert!(stride >= 8);
        assert!(is_aligned(stride * 4, METAL_CACHE_LINE_BYTES));
    }

    #[test]
    fn test_stride_for_coalesced_access_exact_cache_line() {
        // 128 bytes / 4 = 32 f32 elements per cache line
        let stride = coalesced_stride_f32(32);
        assert_eq!(stride, 32);
    }

    #[test]
    fn test_stride_for_coalesced_access_not_aligned() {
        let stride = coalesced_stride_f32(33);
        assert_eq!(stride, 64); // next cache-line aligned
    }

    #[test]
    fn test_misaligned_offset_detected() {
        let layout = compute_buffer_layout_with_offset(100, 4, METAL_OPTIMAL_ALIGNMENT, 3);
        assert!(is_aligned(layout.offset, METAL_OPTIMAL_ALIGNMENT));
        assert_ne!(layout.offset, 3); // was corrected
    }

    #[test]
    fn test_aligned_offset_unchanged() {
        let layout = compute_buffer_layout_with_offset(100, 4, METAL_OPTIMAL_ALIGNMENT, 256);
        assert_eq!(layout.offset, 256);
    }

    #[test]
    fn test_round_to_page_boundary() {
        assert_eq!(round_to_page(1), METAL_PAGE_SIZE);
        assert_eq!(round_to_page(4096), METAL_PAGE_SIZE);
        assert_eq!(round_to_page(4097), 2 * METAL_PAGE_SIZE);
    }

    #[test]
    fn test_large_buffer_alignment_preserved() {
        let layout = compute_buffer_layout(1_000_000, 4, METAL_PAGE_SIZE);
        assert!(is_aligned(layout.size, METAL_PAGE_SIZE));
        assert!(layout.size >= 4_000_000);
    }

    #[test]
    fn test_zero_size_buffer_alignment() {
        let layout = compute_buffer_layout(0, 4, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 0);
        assert!(is_aligned(layout.size, METAL_OPTIMAL_ALIGNMENT));
    }
}

// ═══════════════════════════════════════════════════════════════════
// B. Threadgroup Memory Layout Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod threadgroup_memory_layout {
    use super::*;

    #[test]
    fn test_8x8_tile_total_threads() {
        let tg = ThreadgroupConfig { width: 8, height: 8, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 64);
    }

    #[test]
    fn test_16x16_tile_total_threads() {
        let tg = ThreadgroupConfig { width: 16, height: 16, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 256);
    }

    #[test]
    fn test_32x32_tile_total_threads() {
        let tg = ThreadgroupConfig { width: 32, height: 32, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 1024);
    }

    #[test]
    fn test_32x32_tile_at_max() {
        let tg = ThreadgroupConfig { width: 32, height: 32, depth: 1 };
        assert!(tg.width * tg.height * tg.depth <= METAL_MAX_THREADGROUP_SIZE);
    }

    #[test]
    fn test_tile_exceeds_max_threadgroup() {
        let tg = ThreadgroupConfig { width: 64, height: 32, depth: 1 };
        assert!(tg.width * tg.height * tg.depth > METAL_MAX_THREADGROUP_SIZE);
    }

    #[test]
    fn test_bank_conflict_stride_1_no_conflicts() {
        // Stride-1: thread i accesses element i → no conflicts
        let conflicts = bank_conflict_count(1, 32);
        assert_eq!(conflicts, 0);
    }

    #[test]
    fn test_bank_conflict_stride_32_all_conflict() {
        // Stride-32: every thread maps to bank 0
        let conflicts = bank_conflict_count(32, 32);
        assert_eq!(conflicts, 31); // all 32 threads in bank 0
    }

    #[test]
    fn test_bank_conflict_stride_2_half_conflict() {
        let conflicts = bank_conflict_count(2, 32);
        // Stride-2: threads 0,16→bank 0; 1,17→bank 2; etc
        // Each bank gets exactly 2 hits → 16 banks × 1 conflict
        assert_eq!(conflicts, 16);
    }

    #[test]
    fn test_bank_conflict_stride_16() {
        let conflicts = bank_conflict_count(16, 32);
        // Stride-16: threads land in alternating banks
        assert!(conflicts > 0);
    }

    #[test]
    fn test_padding_eliminates_stride_32_conflict() {
        let padded = threadgroup_memory_padded_stride(32, 4); // 32 f32 = 128B
        // 128B / 4 = 32 banks → needs padding
        assert_eq!(padded, 132); // 128 + 4 bytes padding
        let new_stride_elements = padded / 4;
        let conflicts = bank_conflict_count(new_stride_elements, 32);
        assert_eq!(conflicts, 0);
    }

    #[test]
    fn test_no_padding_when_not_power_of_banks() {
        // 33 f32 = 132 bytes → stride_in_banks = 33, 33%32 != 0
        let padded = threadgroup_memory_padded_stride(33, 4);
        assert_eq!(padded, 33 * 4); // unchanged
    }

    #[test]
    fn test_8x8_tile_shared_memory_size() {
        // 8×8 tile of f32 with padding
        let stride = threadgroup_memory_padded_stride(8, 4);
        let total = stride as usize * 8;
        assert!(total <= 4096); // fits in typical threadgroup memory
    }

    #[test]
    fn test_16x16_tile_shared_memory_size() {
        let stride = threadgroup_memory_padded_stride(16, 4);
        let total = stride as usize * 16;
        assert!(total <= 16384);
    }

    #[test]
    fn test_32x32_tile_shared_memory_size() {
        let stride = threadgroup_memory_padded_stride(32, 4);
        let total = stride as usize * 32;
        assert!(total <= 32768); // 32 KB limit
    }

    #[test]
    fn test_bank_conflict_odd_stride_no_conflict() {
        // Odd strides avoid power-of-two bank aliasing
        let conflicts = bank_conflict_count(31, 32);
        assert_eq!(conflicts, 0);
    }

    #[test]
    fn test_threadgroup_depth_3d() {
        let tg = ThreadgroupConfig { width: 8, height: 8, depth: 4 };
        assert_eq!(tg.width * tg.height * tg.depth, 256);
        assert!(tg.width * tg.height * tg.depth <= METAL_MAX_THREADGROUP_SIZE);
    }

    #[test]
    fn test_padding_stride_64_elements() {
        let padded = threadgroup_memory_padded_stride(64, 4);
        // 64*4=256B, 256/4=64 banks-equiv, 64%32==0 → needs padding
        assert_eq!(padded, 260);
    }
}

// ═══════════════════════════════════════════════════════════════════
// C. Coalesced Access Pattern Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod coalesced_access_patterns {
    use super::*;

    #[test]
    fn test_row_major_stride1_perfect_coalescing() {
        let eff = coalescing_efficiency_row_major(128, 32);
        assert!((eff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_row_major_half_line_efficiency() {
        // 16 threads → 16 floats = 64 bytes → half a 128-byte line
        let eff = coalescing_efficiency_row_major(128, 16);
        assert!((eff - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_row_major_single_thread_low_efficiency() {
        let eff = coalescing_efficiency_row_major(128, 1);
        let expected = 4.0 / METAL_CACHE_LINE_BYTES as f32;
        assert!((eff - expected).abs() < 1e-5);
    }

    #[test]
    fn test_column_major_poor_coalescing() {
        // Column-major with large stride → each thread hits a
        // different cache line.
        let eff = coalescing_efficiency_column_major(1024, 1024, 32);
        assert!(eff < 0.2);
    }

    #[test]
    fn test_column_major_small_stride_better() {
        // Stride of 1 element → threads are close together
        let eff = coalescing_efficiency_column_major(1024, 1, 32);
        assert!(eff > 0.5);
    }

    #[test]
    fn test_row_major_beats_column_major() {
        let row_eff = coalescing_efficiency_row_major(256, 32);
        let col_eff = coalescing_efficiency_column_major(256, 256, 32);
        assert!(row_eff > col_eff);
    }

    #[test]
    fn test_sequential_offsets_perfect_coalescing() {
        let offsets: Vec<usize> = (0..32).collect();
        let m = analyse_access_pattern(&offsets);
        assert!((m.efficiency - 1.0).abs() < 1e-5);
        assert_eq!(m.transactions, 1); // all in one cache line
    }

    #[test]
    fn test_strided_offsets_poor_coalescing() {
        // Each offset in a different cache line (stride = 32 elements)
        let offsets: Vec<usize> = (0..32).map(|i| i * 32).collect();
        let m = analyse_access_pattern(&offsets);
        assert!(m.efficiency < 0.2);
        assert!(m.transactions > 1);
    }

    #[test]
    fn test_empty_offsets() {
        let m = analyse_access_pattern(&[]);
        assert_eq!(m.efficiency, 0.0);
        assert_eq!(m.transactions, 0);
    }

    #[test]
    fn test_single_offset() {
        let m = analyse_access_pattern(&[0]);
        assert_eq!(m.transactions, 1);
    }

    #[test]
    fn test_two_cache_line_boundary() {
        // 32 elements at offset 0..31 = 1 line; 32 at 32..63 = 2nd line
        let offsets: Vec<usize> = (0..64).collect();
        let m = analyse_access_pattern(&offsets);
        assert_eq!(m.transactions, 2);
    }

    #[test]
    fn test_random_scattered_offsets() {
        let offsets = vec![0, 1000, 2000, 3000, 50, 1050, 2050, 3050];
        let m = analyse_access_pattern(&offsets);
        assert!(m.transactions >= 4);
        assert!(m.efficiency < 0.5);
    }

    #[test]
    fn test_coalescing_efficiency_full_simd_group() {
        let eff = coalescing_efficiency_row_major(1024, 32);
        assert!((eff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_coalescing_64_threads_two_lines() {
        // 64 f32 = 256 bytes → exactly 2 cache lines
        let eff = coalescing_efficiency_row_major(1024, 64);
        assert!((eff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_coalescing_non_power_of_two_threads() {
        let eff = coalescing_efficiency_row_major(1024, 24);
        // 24 floats = 96 bytes in 128-byte line → 75%
        assert!((eff - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_matrix_row_access_coalesced() {
        // Simulating a 64×64 matrix row-major; thread i reads col i
        let row = 3usize;
        let cols = 64;
        let offsets: Vec<usize> = (0..32).map(|c| row * cols + c).collect();
        let m = analyse_access_pattern(&offsets);
        assert_eq!(m.transactions, 1); // all within one line
    }

    #[test]
    fn test_matrix_column_access_uncoalesced() {
        // Thread i reads row i, same column → stride = cols
        let col = 3usize;
        let cols = 256;
        let offsets: Vec<usize> = (0..32).map(|r| r * cols + col).collect();
        let m = analyse_access_pattern(&offsets);
        // Each row is 1024 bytes apart → 32 distinct lines
        assert!(m.transactions >= 16);
    }

    #[test]
    fn test_broadcast_read_single_address() {
        // All threads read the same element (broadcast)
        let offsets: Vec<usize> = vec![42; 32];
        let m = analyse_access_pattern(&offsets);
        assert_eq!(m.transactions, 1);
        assert!((m.efficiency - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_coalescing_with_gap() {
        // 16 elements, gap of 16, then 16 more
        let mut offsets: Vec<usize> = (0..16).collect();
        offsets.extend(48..64);
        let m = analyse_access_pattern(&offsets);
        assert_eq!(m.transactions, 2);
    }

    #[test]
    fn test_row_major_256_width_efficiency() {
        let eff = coalescing_efficiency_row_major(256, 32);
        assert!((eff - 1.0).abs() < 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════
// D. Buffer Binding Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod buffer_binding {
    use super::*;

    #[test]
    fn test_f32_buffer_layout() {
        let layout = compute_buffer_layout(1024, 4, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 4096);
        assert!(is_aligned(layout.size, METAL_OPTIMAL_ALIGNMENT));
    }

    #[test]
    fn test_f16_buffer_layout() {
        let layout = compute_buffer_layout(1024, 2, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 2048);
    }

    #[test]
    fn test_i8_buffer_layout() {
        let layout = compute_buffer_layout(1024, 1, METAL_OPTIMAL_ALIGNMENT);
        assert_eq!(layout.size, 1024);
    }

    #[test]
    fn test_i2_packed_buffer_layout() {
        // 4 i2 values per byte
        let num_elements = 1024;
        let bytes = (num_elements + 3) / 4;
        let layout = compute_buffer_layout(bytes, 1, METAL_OPTIMAL_ALIGNMENT);
        assert!(is_aligned(layout.size, METAL_OPTIMAL_ALIGNMENT));
        assert_eq!(layout.size, 256);
    }

    #[test]
    fn test_buffer_offset_alignment_f32() {
        let layout = compute_buffer_layout_with_offset(256, 4, METAL_OPTIMAL_ALIGNMENT, 100);
        assert!(is_aligned(layout.offset, METAL_OPTIMAL_ALIGNMENT));
        assert_eq!(layout.offset, 256);
    }

    #[test]
    fn test_buffer_offset_alignment_f16() {
        let layout = compute_buffer_layout_with_offset(512, 2, METAL_OPTIMAL_ALIGNMENT, 50);
        assert!(is_aligned(layout.offset, METAL_OPTIMAL_ALIGNMENT));
    }

    #[test]
    fn test_buffer_offset_alignment_i8() {
        let layout = compute_buffer_layout_with_offset(1000, 1, METAL_MIN_BUFFER_ALIGNMENT, 7);
        assert!(is_aligned(layout.offset, METAL_MIN_BUFFER_ALIGNMENT));
    }

    #[test]
    fn test_buffer_size_page_rounding_small() {
        let size = round_to_page(100);
        assert_eq!(size, METAL_PAGE_SIZE);
    }

    #[test]
    fn test_buffer_size_page_rounding_exact() {
        let size = round_to_page(4096);
        assert_eq!(size, 4096);
    }

    #[test]
    fn test_buffer_size_page_rounding_large() {
        let size = round_to_page(10000);
        assert_eq!(size, 12288); // 3 pages
    }

    #[test]
    fn test_argument_buffer_multiple_bindings() {
        // Simulate 3 buffer bindings packed in an argument buffer
        let layouts: Vec<MetalBufferLayout> = vec![
            compute_buffer_layout(1024, 4, METAL_OPTIMAL_ALIGNMENT),
            compute_buffer_layout(2048, 2, METAL_OPTIMAL_ALIGNMENT),
            compute_buffer_layout(512, 1, METAL_OPTIMAL_ALIGNMENT),
        ];
        // Each binding should be independently aligned
        for l in &layouts {
            assert!(is_aligned(l.size, METAL_OPTIMAL_ALIGNMENT));
        }
        // Total size when concatenated with alignment gaps
        let mut total = 0usize;
        for l in &layouts {
            total = align_up(total, l.alignment) + l.size;
        }
        assert!(is_aligned(total, METAL_OPTIMAL_ALIGNMENT));
    }

    #[test]
    fn test_subbuffer_offset_within_page() {
        let page_buffer = compute_buffer_layout(4096, 1, METAL_PAGE_SIZE);
        assert_eq!(page_buffer.size, METAL_PAGE_SIZE);
        // Sub-allocation at offset 256 within the page
        let sub = compute_buffer_layout_with_offset(64, 4, METAL_OPTIMAL_ALIGNMENT, 256);
        assert_eq!(sub.offset, 256);
        assert!(sub.offset + sub.size <= page_buffer.size);
    }

    #[test]
    fn test_i2_buffer_packing_alignment() {
        // 4096 i2 values → 1024 bytes → exactly 1 page boundary check
        let bytes = 4096 / 4;
        let layout = compute_buffer_layout(bytes, 1, METAL_PAGE_SIZE);
        assert_eq!(layout.size, METAL_PAGE_SIZE);
    }

    #[test]
    fn test_f32_large_tensor_page_aligned() {
        // 1M floats = 4 MB
        let layout = compute_buffer_layout(1_000_000, 4, METAL_PAGE_SIZE);
        assert!(is_aligned(layout.size, METAL_PAGE_SIZE));
        assert!(layout.size >= 4_000_000);
    }

    #[test]
    fn test_zero_offset_is_aligned() {
        let layout = compute_buffer_layout_with_offset(100, 4, METAL_OPTIMAL_ALIGNMENT, 0);
        assert_eq!(layout.offset, 0);
        assert!(is_aligned(layout.offset, METAL_OPTIMAL_ALIGNMENT));
    }
}

// ═══════════════════════════════════════════════════════════════════
// E. Dispatch Sizing for Coalescing Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod dispatch_sizing {
    use super::*;

    #[test]
    fn test_threadgroup_32_for_narrow_tensor() {
        let tg = compute_threadgroup_for_coalescing(32, 1024);
        assert_eq!(tg.width, 32);
        assert!(tg.width * tg.height <= METAL_MAX_THREADGROUP_SIZE);
    }

    #[test]
    fn test_threadgroup_width_clamped_to_tensor() {
        let tg = compute_threadgroup_for_coalescing(16, 512);
        assert_eq!(tg.width, 16);
    }

    #[test]
    fn test_threadgroup_height_fills_budget() {
        let tg = compute_threadgroup_for_coalescing(32, 1024);
        // 1024 / 32 = 32 rows
        assert_eq!(tg.height, 32);
        assert_eq!(tg.width * tg.height, METAL_MAX_THREADGROUP_SIZE);
    }

    #[test]
    fn test_threadgroup_height_clamped_to_tensor() {
        let tg = compute_threadgroup_for_coalescing(32, 4);
        assert_eq!(tg.height, 4);
    }

    #[test]
    fn test_threads_per_threadgroup_32() {
        let tg = ThreadgroupConfig { width: 32, height: 1, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 32);
    }

    #[test]
    fn test_threads_per_threadgroup_64() {
        let tg = ThreadgroupConfig { width: 32, height: 2, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 64);
    }

    #[test]
    fn test_threads_per_threadgroup_128() {
        let tg = ThreadgroupConfig { width: 32, height: 4, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 128);
    }

    #[test]
    fn test_threads_per_threadgroup_256() {
        let tg = ThreadgroupConfig { width: 16, height: 16, depth: 1 };
        assert_eq!(tg.width * tg.height * tg.depth, 256);
    }

    #[test]
    fn test_grid_size_exact_divisible() {
        let tg = ThreadgroupConfig { width: 32, height: 8, depth: 1 };
        let d = compute_dispatch([128, 64, 1], &tg);
        assert_eq!(d.grid, [4, 8, 1]);
    }

    #[test]
    fn test_grid_size_with_remainder() {
        let tg = ThreadgroupConfig { width: 32, height: 8, depth: 1 };
        let d = compute_dispatch([100, 50, 1], &tg);
        assert_eq!(d.grid, [4, 7, 1]); // ceil(100/32)=4, ceil(50/8)=7
    }

    #[test]
    fn test_grid_size_single_threadgroup() {
        let tg = ThreadgroupConfig { width: 32, height: 32, depth: 1 };
        let d = compute_dispatch([16, 16, 1], &tg);
        assert_eq!(d.grid, [1, 1, 1]);
    }

    #[test]
    fn test_dispatch_preserves_threadgroup() {
        let tg = ThreadgroupConfig { width: 32, height: 8, depth: 1 };
        let d = compute_dispatch([256, 128, 1], &tg);
        assert_eq!(d.threadgroup, [32, 8, 1]);
    }

    #[test]
    fn test_3d_dispatch() {
        let tg = ThreadgroupConfig { width: 16, height: 8, depth: 4 };
        let d = compute_dispatch([64, 32, 16], &tg);
        assert_eq!(d.grid, [4, 4, 4]);
    }

    #[test]
    fn test_batch_dispatch_depth() {
        let batch = 8u32;
        let tg = ThreadgroupConfig { width: 32, height: 8, depth: 1 };
        let d = compute_dispatch([128, 64, batch], &tg);
        assert_eq!(d.grid[2], batch);
    }

    #[test]
    fn test_total_threads_cover_tensor() {
        let tensor = [100u32, 50, 1];
        let tg = compute_threadgroup_for_coalescing(tensor[0], tensor[1]);
        let d = compute_dispatch(tensor, &tg);
        let total_x = d.grid[0] * d.threadgroup[0];
        let total_y = d.grid[1] * d.threadgroup[1];
        assert!(total_x >= tensor[0]);
        assert!(total_y >= tensor[1]);
    }

    #[test]
    fn test_simdgroup_aligned_width() {
        // Width should be multiple of SIMD group size for coalescing
        let tg = compute_threadgroup_for_coalescing(256, 256);
        assert_eq!(tg.width % METAL_SIMDGROUP_SIZE, 0);
    }

    #[test]
    fn test_small_tensor_dispatch() {
        let tg = compute_threadgroup_for_coalescing(4, 4);
        assert_eq!(tg.width, 4);
        assert_eq!(tg.height, 4);
        let d = compute_dispatch([4, 4, 1], &tg);
        assert_eq!(d.grid, [1, 1, 1]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// F. Memory Access Pattern Analysis Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod access_pattern_analysis {
    use super::*;

    // ── Sequential vs Strided ───────────────────────────────────────

    #[test]
    fn test_sequential_access_classified() {
        let offsets: Vec<usize> = (0..32).collect();
        let m = analyse_access_pattern(&offsets);
        assert!((m.efficiency - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_strided_access_classified() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 64).collect();
        let m = analyse_access_pattern(&offsets);
        assert!(m.efficiency < 0.15);
    }

    #[test]
    fn test_stride2_moderate_efficiency() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 2).collect();
        let m = analyse_access_pattern(&offsets);
        // 64 elements span → 2 cache lines used
        assert!(m.efficiency > 0.3);
        assert!(m.efficiency < 1.0);
    }

    #[test]
    fn test_stride4_efficiency() {
        let offsets: Vec<usize> = (0..32).map(|i| i * 4).collect();
        let m = analyse_access_pattern(&offsets);
        assert!(m.transactions >= 3);
    }

    // ── Transpose Operations ────────────────────────────────────────

    #[test]
    fn test_transpose_read_pattern() {
        // Reading column 0 of a 64-wide matrix row by row
        let cols = 64usize;
        let offsets: Vec<usize> = (0..32).map(|r| r * cols).collect();
        let m = analyse_access_pattern(&offsets);
        // Each row start is 256 bytes apart → many cache lines
        assert!(m.transactions >= 8);
    }

    #[test]
    fn test_transpose_write_pattern_tiled() {
        // 8×8 tile transpose: read tile from rows, write to columns
        // For a 64-wide matrix, read 8 consecutive elements per row
        let cols = 64usize;
        let read_offsets: Vec<usize> =
            (0..8).flat_map(|r| (0..8).map(move |c| r * cols + c)).collect();
        let m = analyse_access_pattern(&read_offsets);
        // 8 rows × 8 cols, rows ~256B apart → read ≤8 lines
        assert!(m.transactions <= 8);
    }

    #[test]
    fn test_transpose_small_matrix_coalesced_read() {
        // 32-wide matrix: reading 32 consecutive col elements = 1 line
        let cols = 32usize;
        let offsets: Vec<usize> = (0..32).map(|c| 0 * cols + c).collect();
        let m = analyse_access_pattern(&offsets);
        assert_eq!(m.transactions, 1);
    }

    #[test]
    fn test_transpose_output_column_scattered() {
        // Writing column 0 of a 256-wide output → very scattered
        let cols = 256usize;
        let offsets: Vec<usize> = (0..32).map(|r| r * cols + 0).collect();
        let m = analyse_access_pattern(&offsets);
        assert!(m.transactions >= 16);
    }

    // ── Reduction Operations ────────────────────────────────────────

    #[test]
    fn test_reduction_sum_coalesced_load() {
        let eff = reduction_coalescing_efficiency(1024, 256);
        assert!((eff - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduction_sum_partial_line() {
        let eff = reduction_coalescing_efficiency(1024, 16);
        assert!((eff - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_reduction_max_same_as_sum() {
        // Coalescing pattern is the same for sum and max reductions
        let eff_sum = reduction_coalescing_efficiency(1024, 128);
        let eff_max = reduction_coalescing_efficiency(1024, 128);
        assert!((eff_sum - eff_max).abs() < 1e-6);
    }

    #[test]
    fn test_reduction_small_input() {
        let eff = reduction_coalescing_efficiency(8, 32);
        // Only 8 useful elements loaded by 32 threads
        let expected = 8.0 * 4.0 / METAL_CACHE_LINE_BYTES as f32;
        assert!((eff - expected).abs() < 1e-5);
    }

    #[test]
    fn test_reduction_single_thread() {
        let eff = reduction_coalescing_efficiency(1024, 1);
        let expected = 4.0 / METAL_CACHE_LINE_BYTES as f32;
        assert!((eff - expected).abs() < 1e-5);
    }

    #[test]
    fn test_reduction_full_cache_line() {
        // 32 floats = 128 bytes = exactly 1 cache line
        let eff = reduction_coalescing_efficiency(1024, 32);
        assert!((eff - 1.0).abs() < 1e-5);
    }

    // ── Batch Matrix Multiply ───────────────────────────────────────

    #[test]
    fn test_batch_matmul_single_tile() {
        let txns = batch_matmul_transactions(8, 8, 8, 8, 8, 8, 1);
        // Loads: 8*8 + 8*8 = 128 elements → 4 lines
        assert!(txns > 0);
    }

    #[test]
    fn test_batch_matmul_multi_tile() {
        let txns = batch_matmul_transactions(64, 64, 64, 8, 8, 8, 1);
        // 8×8×8 tiling of 64×64×64 → many more transactions
        let single = batch_matmul_transactions(8, 8, 8, 8, 8, 8, 1);
        assert!(txns > single);
    }

    #[test]
    fn test_batch_matmul_batch_scales_linearly() {
        let txns_1 = batch_matmul_transactions(32, 32, 32, 8, 8, 8, 1);
        let txns_4 = batch_matmul_transactions(32, 32, 32, 8, 8, 8, 4);
        assert_eq!(txns_4, txns_1 * 4);
    }

    #[test]
    fn test_batch_matmul_larger_tile_fewer_transactions() {
        let txns_small = batch_matmul_transactions(64, 64, 64, 8, 8, 8, 1);
        let txns_large = batch_matmul_transactions(64, 64, 64, 16, 16, 16, 1);
        // Larger tiles → fewer tile iterations → fewer transactions
        assert!(txns_large < txns_small);
    }

    #[test]
    fn test_batch_matmul_asymmetric_tiles() {
        let txns = batch_matmul_transactions(128, 64, 32, 16, 8, 8, 1);
        assert!(txns > 0);
    }

    #[test]
    fn test_batch_matmul_non_divisible_dims() {
        let txns = batch_matmul_transactions(100, 50, 70, 16, 16, 16, 1);
        assert!(txns > 0);
    }

    // ── Mixed patterns ──────────────────────────────────────────────

    #[test]
    fn test_interleaved_access() {
        // Even threads read from region A, odd from region B
        let offsets: Vec<usize> =
            (0..32).map(|i| if i % 2 == 0 { i / 2 } else { 1000 + i / 2 }).collect();
        let m = analyse_access_pattern(&offsets);
        assert!(m.transactions >= 2);
    }

    #[test]
    fn test_gather_pattern_random_indices() {
        // Simulating a gather with scattered indices
        let indices = [
            0, 500, 1000, 42, 999, 3, 750, 200, 128, 64, 900, 100, 450, 800, 350, 600, 50, 150,
            250, 700, 850, 950, 400, 550, 650, 300, 75, 125, 425, 525, 775, 925,
        ];
        let m = analyse_access_pattern(&indices);
        // Widely scattered → many transactions
        assert!(m.transactions >= 16);
        assert!(m.efficiency < 0.3);
    }

    #[test]
    fn test_tiled_access_pattern_8x8() {
        // Access an 8×8 tile from a 128-wide matrix
        let cols = 128usize;
        let tile_row = 2usize;
        let tile_col = 3usize;
        let offsets: Vec<usize> = (0..8)
            .flat_map(|r| {
                let row = tile_row * 8 + r;
                (0..8).map(move |c| row * cols + tile_col * 8 + c)
            })
            .collect();
        let m = analyse_access_pattern(&offsets);
        // 8 consecutive elements per row; rows 512B apart
        assert!(m.transactions <= 8);
    }

    #[test]
    fn test_vectorized_load_pattern() {
        // float4 loads: each thread reads 4 consecutive floats
        let offsets: Vec<usize> = (0..32).flat_map(|t| (0..4).map(move |v| t * 4 + v)).collect();
        let m = analyse_access_pattern(&offsets);
        // 128 consecutive elements → 4 cache lines
        assert_eq!(m.transactions, 4);
    }

    #[test]
    fn test_diagonal_access_pattern() {
        // Thread i reads element (i, i) in a matrix
        let cols = 64usize;
        let offsets: Vec<usize> = (0..32).map(|i| i * cols + i).collect();
        let m = analyse_access_pattern(&offsets);
        // Diagonal → scattered across many lines
        assert!(m.transactions >= 8);
    }
}

// ═══════════════════════════════════════════════════════════════════
// G. Additional Edge Cases & Property Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod additional_edge_cases {
    use super::*;

    #[test]
    fn test_align_up_identity_for_aligned() {
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(0, 16), 0);
    }

    #[test]
    fn test_align_up_rounds_correctly() {
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    #[test]
    fn test_is_aligned_basic() {
        assert!(is_aligned(0, 16));
        assert!(is_aligned(16, 16));
        assert!(!is_aligned(15, 16));
        assert!(is_aligned(4096, 4096));
    }

    #[test]
    fn test_coalescing_metrics_bank_conflicts_stride1() {
        // Sequential 32-thread access → 0 bank conflicts
        let offsets: Vec<usize> = (0..32).collect();
        let m = analyse_access_pattern(&offsets);
        assert_eq!(m.bank_conflicts, 0);
    }

    #[test]
    fn test_dispatch_covers_minimum_1x1() {
        let tg = ThreadgroupConfig { width: 1, height: 1, depth: 1 };
        let d = compute_dispatch([1, 1, 1], &tg);
        assert_eq!(d.grid, [1, 1, 1]);
    }

    #[test]
    fn test_large_dispatch_grid() {
        let tg = ThreadgroupConfig { width: 32, height: 1, depth: 1 };
        let d = compute_dispatch([100_000, 1, 1], &tg);
        assert_eq!(d.grid[0], 3125);
    }

    #[test]
    fn test_threadgroup_memory_padded_stride_f16() {
        // 64 f16 values = 128 bytes = 32 banks → needs padding
        let padded = threadgroup_memory_padded_stride(64, 2);
        assert_eq!(padded, 132); // 128 + 4
    }

    #[test]
    fn test_threadgroup_memory_padded_stride_i8() {
        // 128 i8 values = 128 bytes = 32 banks → needs padding
        let padded = threadgroup_memory_padded_stride(128, 1);
        assert_eq!(padded, 132);
    }

    #[test]
    fn test_buffer_layout_equality() {
        let a = compute_buffer_layout(100, 4, 256);
        let b = compute_buffer_layout(100, 4, 256);
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_alignment_different_layout() {
        let a = compute_buffer_layout(100, 4, 16);
        let b = compute_buffer_layout(100, 4, 256);
        assert_ne!(a.size, b.size);
    }

    #[test]
    fn test_coalescing_efficiency_never_exceeds_one() {
        for threads in [1, 2, 4, 8, 16, 32, 64, 128] {
            let eff = coalescing_efficiency_row_major(1024, threads);
            assert!(eff <= 1.0 + 1e-6, "threads={threads} eff={eff}");
        }
    }

    #[test]
    fn test_reduction_efficiency_never_exceeds_one() {
        for tg in [1, 16, 32, 64, 128, 256] {
            let eff = reduction_coalescing_efficiency(4096, tg);
            assert!(eff <= 1.0 + 1e-6, "tg={tg} eff={eff}");
        }
    }

    #[test]
    fn test_bank_conflict_count_never_negative() {
        for stride in 1..=64 {
            let c = bank_conflict_count(stride, 32);
            // c is u32, can't be negative; just ensure no panic
            let _ = c;
        }
    }

    #[test]
    fn test_threadgroup_for_coalescing_respects_max() {
        for (w, h) in [(1024, 1024), (1, 4096), (4096, 1)] {
            let tg = compute_threadgroup_for_coalescing(w, h);
            assert!(
                tg.width * tg.height * tg.depth <= METAL_MAX_THREADGROUP_SIZE,
                "w={w} h={h} tg={tg:?}"
            );
        }
    }

    #[test]
    fn test_page_rounding_idempotent() {
        let v = round_to_page(4096);
        assert_eq!(round_to_page(v), v);
    }

    #[test]
    fn test_batch_matmul_zero_batch_zero_transactions() {
        let txns = batch_matmul_transactions(32, 32, 32, 8, 8, 8, 0);
        assert_eq!(txns, 0);
    }
}
