#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal buffer alignment and compute dispatch tests for Apple Silicon.
//!
//! Validates Metal GPU buffer alignment requirements, threadgroup sizing,
//! and memory layout constraints critical for correct kernel dispatch.
//! Tests exercise the public API from `bitnet_kernels::metal_compute`.

#![cfg(feature = "metal")]

use bitnet_kernels::metal_compute::{
    DispatchDimensions, MAX_DISPATCH_DIM, METAL_BUFFER_ALIGNMENT, METAL_MAX_WORKGROUP_SIZE,
    MemoryArchitecture, MetalComputePipeline, MetalConfigError, WorkgroupSize, align_buffer_size,
    is_aligned,
};

// ── Helper functions ────────────────────────────────────────────────

/// Round `size` up to the next multiple of `alignment`.
/// `alignment` must be a power of two.
fn align_to(size: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be power of 2");
    let mask = alignment - 1;
    (size + mask) & !mask
}

/// Compute threadgroup size and dispatch group count for a 1-D problem.
/// Returns `(threads_per_group, num_groups)`.
fn compute_threadgroup_size(total: usize, max_per_group: usize) -> (usize, usize) {
    if total == 0 {
        return (0, 0);
    }
    let group_size = total.min(max_per_group);
    let num_groups = total.div_ceil(group_size);
    (group_size, num_groups)
}

/// Compute the number of dispatch groups needed to cover `total` items
/// with groups of `group_size`.
fn compute_dispatch_groups(total: usize, group_size: usize) -> usize {
    if group_size == 0 {
        return 0;
    }
    total.div_ceil(group_size)
}

// ── Apple Silicon constants ─────────────────────────────────────────

/// Metal page size on Apple Silicon (16 KiB).
const PAGE_ALIGNMENT: usize = 16384;

/// SIMD group width on Apple Silicon GPUs (M1/M2/M3/M4).
const SIMD_GROUP_WIDTH: u32 = 32;

/// Maximum shared (threadgroup) memory per threadgroup on Apple Silicon.
const MAX_SHARED_MEMORY_PER_THREADGROUP: usize = 32 * 1024; // 32 KiB

/// Minimum uniform buffer offset alignment (Metal spec).
const UNIFORM_BUFFER_OFFSET_ALIGNMENT: usize = 256;

/// Maximum buffer length for Metal on Apple Silicon (practical limit).
/// Apple Silicon unified memory can address up to the full physical RAM,
/// but Metal caps individual buffer allocations at ~256 TB virtual.
/// For testing we use a conservative 256 GiB boundary.
const MAX_BUFFER_LENGTH_APPROX: usize = 256 * 1024 * 1024 * 1024;

/// Texture row alignment requirement (Metal best-practice: 256 bytes).
const TEXTURE_ROW_ALIGNMENT: usize = 256;

/// Resource heap alignment on Apple Silicon.
const RESOURCE_HEAP_ALIGNMENT: usize = 256;

// ═══════════════════════════════════════════════════════════════════════
// 1. Buffer alignment (256-byte minimum)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn buffer_alignment_constant_is_256() {
    assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
    assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two());
}

#[test]
fn align_buffer_size_zero_returns_zero() {
    assert_eq!(align_buffer_size(0), 0);
}

#[test]
fn align_buffer_size_exact_multiple_unchanged() {
    for multiple in 1..=16 {
        let size = METAL_BUFFER_ALIGNMENT * multiple;
        assert_eq!(
            align_buffer_size(size),
            size,
            "already-aligned size {size} should be unchanged"
        );
    }
}

#[test]
fn align_buffer_size_rounds_up_to_next_boundary() {
    assert_eq!(align_buffer_size(1), 256);
    assert_eq!(align_buffer_size(128), 256);
    assert_eq!(align_buffer_size(255), 256);
    assert_eq!(align_buffer_size(257), 512);
    assert_eq!(align_buffer_size(500), 512);
    assert_eq!(align_buffer_size(513), 768);
}

#[test]
fn align_buffer_size_matches_align_to_helper() {
    for size in [0, 1, 127, 128, 255, 256, 257, 511, 512, 1023, 1024, 4096, 65535] {
        assert_eq!(
            align_buffer_size(size),
            align_to(size, METAL_BUFFER_ALIGNMENT),
            "mismatch for size {size}"
        );
    }
}

#[test]
fn is_aligned_accepts_multiples_of_256() {
    for i in 0..=32 {
        assert!(is_aligned(i * 256), "offset {} should be aligned", i * 256);
    }
}

#[test]
fn is_aligned_rejects_non_multiples() {
    for offset in [1, 2, 4, 8, 16, 32, 64, 128, 255] {
        assert!(!is_aligned(offset), "offset {offset} should NOT be aligned");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Page alignment for large buffers (16384 bytes)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn page_alignment_is_16k() {
    assert_eq!(PAGE_ALIGNMENT, 16384);
    assert!(PAGE_ALIGNMENT.is_power_of_two());
}

#[test]
fn page_aligned_buffers_are_also_256_aligned() {
    // Any page-aligned address is automatically 256-byte aligned.
    for pages in 1..=8 {
        let addr = pages * PAGE_ALIGNMENT;
        assert!(is_aligned(addr), "page-aligned addr {addr} must satisfy 256-byte rule");
    }
}

#[test]
fn large_buffer_page_alignment() {
    // A 1 MiB allocation should align to pages.
    let one_mib = 1024 * 1024;
    let aligned = align_to(one_mib, PAGE_ALIGNMENT);
    assert_eq!(aligned, one_mib); // 1 MiB is already page-aligned
    assert_eq!(aligned % PAGE_ALIGNMENT, 0);
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Threadgroup size limits (max 1024)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn max_workgroup_size_is_1024() {
    assert_eq!(METAL_MAX_WORKGROUP_SIZE, 1024);
}

#[test]
fn workgroup_exactly_at_limit() {
    // Various factorizations of 1024
    assert!(WorkgroupSize::new(1024, 1, 1).is_ok());
    assert!(WorkgroupSize::new(512, 2, 1).is_ok());
    assert!(WorkgroupSize::new(32, 32, 1).is_ok());
    assert!(WorkgroupSize::new(16, 8, 8).is_ok());
    assert!(WorkgroupSize::new(8, 8, 16).is_ok());
    assert!(WorkgroupSize::new(4, 4, 64).is_ok());

    for wg in [
        WorkgroupSize::new(1024, 1, 1).unwrap(),
        WorkgroupSize::new(32, 32, 1).unwrap(),
        WorkgroupSize::new(16, 8, 8).unwrap(),
    ] {
        assert_eq!(wg.total_threads(), 1024);
    }
}

#[test]
fn workgroup_one_over_limit_rejected() {
    assert!(WorkgroupSize::new(1025, 1, 1).is_err());
    assert!(WorkgroupSize::new(33, 32, 1).is_err()); // 33*32 = 1056
    assert!(WorkgroupSize::new(16, 16, 5).is_err()); // 16*16*5 = 1280
}

#[test]
fn workgroup_zero_dimension_always_rejected() {
    assert_eq!(WorkgroupSize::new(0, 16, 1).unwrap_err(), MetalConfigError::ZeroDimension);
    assert_eq!(WorkgroupSize::new(16, 0, 1).unwrap_err(), MetalConfigError::ZeroDimension);
    assert_eq!(WorkgroupSize::new(16, 16, 0).unwrap_err(), MetalConfigError::ZeroDimension);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. 2-D and 3-D dispatch grid calculations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn dispatch_2d_exact_division() {
    let wg = WorkgroupSize::tile(16).unwrap();
    let d = DispatchDimensions::for_problem((256, 128, 1), &wg).unwrap();
    assert_eq!(d.x, 16); // 256 / 16
    assert_eq!(d.y, 8); // 128 / 16
    assert_eq!(d.z, 1);
}

#[test]
fn dispatch_2d_with_remainder() {
    let wg = WorkgroupSize::tile(16).unwrap();
    let d = DispatchDimensions::for_problem((17, 33, 1), &wg).unwrap();
    assert_eq!(d.x, 2); // ceil(17/16)
    assert_eq!(d.y, 3); // ceil(33/16)
}

#[test]
fn dispatch_3d_batch_dimension() {
    let wg = WorkgroupSize::new(16, 16, 1).unwrap();
    let batch = 4;
    let d = DispatchDimensions::for_problem((64, 64, batch), &wg).unwrap();
    assert_eq!(d.x, 4);
    assert_eq!(d.y, 4);
    assert_eq!(d.z, 4); // batch = 4, wg.z = 1 → ceil(4/1) = 4
}

#[test]
fn dispatch_max_dimension_boundary() {
    let wg = WorkgroupSize::linear(1).unwrap();
    // Exactly at limit
    let d = DispatchDimensions::for_problem((MAX_DISPATCH_DIM, 1, 1), &wg).unwrap();
    assert_eq!(d.x, MAX_DISPATCH_DIM);

    // One over
    let err = DispatchDimensions::for_problem((MAX_DISPATCH_DIM + 1, 1, 1), &wg).unwrap_err();
    assert!(matches!(err, MetalConfigError::DispatchTooLarge { .. }));
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Buffer size rounding (256-byte boundary)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_aligned_buffer_bytes_f32_elements() {
    let p = MetalComputePipeline::new("test");
    // 100 f32 elements = 400 bytes → next 256-boundary = 512
    assert_eq!(p.aligned_buffer_bytes(100, 4), 512);
    // 64 f32 elements = 256 bytes → already aligned
    assert_eq!(p.aligned_buffer_bytes(64, 4), 256);
    // 1 f32 element = 4 bytes → 256
    assert_eq!(p.aligned_buffer_bytes(1, 4), 256);
}

#[test]
fn pipeline_aligned_buffer_bytes_f16_elements() {
    let p = MetalComputePipeline::new("f16");
    // 128 f16 elements = 256 bytes → aligned
    assert_eq!(p.aligned_buffer_bytes(128, 2), 256);
    // 129 f16 elements = 258 bytes → 512
    assert_eq!(p.aligned_buffer_bytes(129, 2), 512);
}

#[test]
fn align_buffer_size_large_values() {
    // 1 MiB - 1 byte → rounds up to 1 MiB
    let almost_mib = 1024 * 1024 - 1;
    let aligned = align_buffer_size(almost_mib);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert!(aligned >= almost_mib);
    assert!(aligned - almost_mib < METAL_BUFFER_ALIGNMENT);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Shared memory limits (32 KB on Apple Silicon)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn shared_memory_limit_is_32k() {
    assert_eq!(MAX_SHARED_MEMORY_PER_THREADGROUP, 32 * 1024);
}

#[test]
fn tile_fits_in_shared_memory() {
    // A 16×16 tile of f32 values = 16*16*4 = 1024 bytes — fits easily.
    let tile_bytes = 16_usize * 16 * 4;
    assert!(
        tile_bytes <= MAX_SHARED_MEMORY_PER_THREADGROUP,
        "16×16 f32 tile ({tile_bytes} bytes) should fit in shared memory"
    );

    // A 32×32 tile of f32 = 4096 bytes — still fits.
    let tile_bytes = 32_usize * 32 * 4;
    assert!(tile_bytes <= MAX_SHARED_MEMORY_PER_THREADGROUP);

    // A 128×64 tile of f32 = 32768 bytes — exactly at limit.
    let tile_bytes = 128_usize * 64 * 4;
    assert_eq!(tile_bytes, MAX_SHARED_MEMORY_PER_THREADGROUP);
}

#[test]
fn oversized_tile_exceeds_shared_memory() {
    // A 128×128 tile of f32 = 65536 bytes — exceeds 32 KiB.
    let tile_bytes = 128_usize * 128 * 4;
    assert!(
        tile_bytes > MAX_SHARED_MEMORY_PER_THREADGROUP,
        "128×128 f32 tile should exceed shared memory limit"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 7. SIMD group width (32 on Apple Silicon)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn simd_group_width_is_32() {
    assert_eq!(SIMD_GROUP_WIDTH, 32);
    assert!(SIMD_GROUP_WIDTH.is_power_of_two());
}

#[test]
fn workgroup_multiple_of_simd_width_is_efficient() {
    // Workgroup sizes that are multiples of 32 avoid partial SIMD groups.
    for multiple in [1, 2, 4, 8, 16, 32] {
        let threads = SIMD_GROUP_WIDTH * multiple;
        if threads <= METAL_MAX_WORKGROUP_SIZE {
            let wg = WorkgroupSize::linear(threads).unwrap();
            assert_eq!(wg.total_threads() % SIMD_GROUP_WIDTH, 0);
        }
    }
}

#[test]
fn non_simd_aligned_workgroup_wastes_lanes() {
    // 33 threads → 2 SIMD groups, second group has 1 active lane out of 32.
    let wg = WorkgroupSize::linear(33).unwrap();
    let full_groups = wg.total_threads() / SIMD_GROUP_WIDTH;
    let remainder = wg.total_threads() % SIMD_GROUP_WIDTH;
    assert_eq!(full_groups, 1);
    assert_eq!(remainder, 1); // 1 wasted partial group
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Maximum buffer size (Apple Silicon unified memory)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn max_buffer_length_is_reasonable() {
    // Must be a multiple of the buffer alignment.
    assert_eq!(MAX_BUFFER_LENGTH_APPROX % METAL_BUFFER_ALIGNMENT, 0);
    // At least 1 GiB.
    assert!(MAX_BUFFER_LENGTH_APPROX >= 1024 * 1024 * 1024);
}

#[test]
fn align_does_not_overflow_on_large_sizes() {
    // Largest size that can be aligned without overflow.
    let large = usize::MAX - (METAL_BUFFER_ALIGNMENT - 1);
    let aligned = align_to(large, METAL_BUFFER_ALIGNMENT);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert!(aligned >= large);
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Uniform buffer offset alignment
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn uniform_buffer_alignment_matches_metal_spec() {
    // Metal requires uniform buffer offsets to be 256-byte aligned.
    assert_eq!(UNIFORM_BUFFER_OFFSET_ALIGNMENT, METAL_BUFFER_ALIGNMENT);
    assert!(UNIFORM_BUFFER_OFFSET_ALIGNMENT.is_power_of_two());
}

#[test]
fn uniform_buffer_offset_validation() {
    for offset in (0..=2048).step_by(256) {
        assert!(
            offset % UNIFORM_BUFFER_OFFSET_ALIGNMENT == 0,
            "offset {offset} should satisfy uniform buffer alignment"
        );
    }
    for bad_offset in [1, 64, 128, 255, 257, 511] {
        assert!(bad_offset % UNIFORM_BUFFER_OFFSET_ALIGNMENT != 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Non-power-of-2 dimension handling
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn dispatch_non_power_of_2_problem_size() {
    let wg = WorkgroupSize::tile(16).unwrap();
    // 100×100 is not a power-of-2 in either dimension.
    let d = DispatchDimensions::for_problem((100, 100, 1), &wg).unwrap();
    assert_eq!(d.x, 7); // ceil(100/16) = 7
    assert_eq!(d.y, 7);
    // Total coverage: 7*16 = 112 >= 100 ✓
    assert!(d.x * wg.x >= 100);
    assert!(d.y * wg.y >= 100);
}

#[test]
fn dispatch_prime_number_dimensions() {
    let wg = WorkgroupSize::linear(32).unwrap();
    // 127 is prime — worst case for tiling.
    let groups = compute_dispatch_groups(127, 32);
    assert_eq!(groups, 4); // ceil(127/32) = 4
    assert!(groups * 32 >= 127);
}

// ═══════════════════════════════════════════════════════════════════════
// 11. Edge cases: zero-size buffers, 1-element dispatches
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn zero_element_buffer_aligns_to_zero() {
    let p = MetalComputePipeline::new("edge");
    assert_eq!(p.aligned_buffer_bytes(0, 4), 0);
}

#[test]
fn single_element_dispatch() {
    let wg = WorkgroupSize::linear(256).unwrap();
    let d = DispatchDimensions::for_problem((1, 1, 1), &wg).unwrap();
    assert_eq!((d.x, d.y, d.z), (1, 1, 1));
}

#[test]
fn single_thread_workgroup() {
    let wg = WorkgroupSize::new(1, 1, 1).unwrap();
    assert_eq!(wg.total_threads(), 1);
    let d = DispatchDimensions::for_problem((100, 1, 1), &wg).unwrap();
    assert_eq!(d.x, 100);
}

#[test]
fn helper_compute_threadgroup_size_zero_total() {
    let (group_size, num_groups) = compute_threadgroup_size(0, 1024);
    assert_eq!(group_size, 0);
    assert_eq!(num_groups, 0);
}

#[test]
fn helper_compute_dispatch_groups_zero_group_size() {
    assert_eq!(compute_dispatch_groups(100, 0), 0);
}

// ═══════════════════════════════════════════════════════════════════════
// 12. Buffer offset alignment for vertex/fragment/compute
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn buffer_offset_alignment_covers_all_stages() {
    // Metal requires 256-byte alignment for buffer offsets in all shader
    // stages: vertex, fragment, and compute.
    let offsets_valid = [0, 256, 512, 1024, 4096];
    let offsets_invalid = [1, 128, 255, 257, 384];

    for &o in &offsets_valid {
        assert!(is_aligned(o), "offset {o} should be valid for all stages");
    }
    for &o in &offsets_invalid {
        assert!(!is_aligned(o), "offset {o} should be invalid");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 13. Maximum threadgroups per grid
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn max_dispatch_dim_is_65535() {
    assert_eq!(MAX_DISPATCH_DIM, 65535);
}

#[test]
fn max_threadgroups_per_grid_3d() {
    // Maximum 3-D grid: 65535 × 65535 × 65535 threadgroups.
    let max_groups_per_axis = MAX_DISPATCH_DIM as u64;
    let max_total = max_groups_per_axis * max_groups_per_axis * max_groups_per_axis;
    // Just verify it's astronomically large (> 2^47).
    assert!(max_total > (1u64 << 47));
}

#[test]
fn dispatch_at_max_dim_boundary() {
    let wg = WorkgroupSize::linear(1).unwrap();
    // At the boundary — should succeed.
    let d = DispatchDimensions::for_problem((65535, 65535, 1), &wg).unwrap();
    assert_eq!(d.x, 65535);
    assert_eq!(d.y, 65535);
}

// ═══════════════════════════════════════════════════════════════════════
// 14. Resource heap alignment
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn resource_heap_alignment_matches_buffer_alignment() {
    // Metal resource heaps require the same 256-byte alignment.
    assert_eq!(RESOURCE_HEAP_ALIGNMENT, METAL_BUFFER_ALIGNMENT);
}

#[test]
fn heap_suballocation_alignment() {
    // When sub-allocating from a heap, each resource must start at a
    // 256-byte boundary.
    let resource_sizes = [100, 300, 1000, 4096];
    let mut offset = 0usize;
    for &size in &resource_sizes {
        assert!(is_aligned(offset), "heap offset {offset} must be aligned");
        let aligned_size = align_buffer_size(size);
        offset += aligned_size;
    }
    // Final offset should also be aligned.
    assert!(is_aligned(offset));
}

// ═══════════════════════════════════════════════════════════════════════
// 15. Texture row alignment
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn texture_row_alignment_is_256() {
    assert_eq!(TEXTURE_ROW_ALIGNMENT, 256);
}

#[test]
fn texture_row_bytes_aligned() {
    // A 1024-pixel row of RGBA8 (4 bytes/pixel) = 4096 bytes → aligned.
    let row_bytes = 1024 * 4;
    assert_eq!(align_to(row_bytes, TEXTURE_ROW_ALIGNMENT), row_bytes);

    // A 100-pixel row of RGBA8 = 400 bytes → rounds to 512.
    let row_bytes = 100 * 4;
    assert_eq!(align_to(row_bytes, TEXTURE_ROW_ALIGNMENT), 512);
}

// ═══════════════════════════════════════════════════════════════════════
// 16. Pipeline integration: matrix dispatch with Metal constraints
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn pipeline_dispatch_non_square_matrix() {
    let p = MetalComputePipeline::new("gemm");
    // 768 × 2048 matrix, default 16×16 tile.
    let d = p.dispatch_for_matrix(768, 2048).unwrap();
    assert_eq!(d.x, 128); // 2048 / 16
    assert_eq!(d.y, 48); // 768 / 16
}

#[test]
fn pipeline_memory_architecture_apple_silicon() {
    let p = MetalComputePipeline::new("test");
    // On Apple Silicon (aarch64-apple-*) this should be Unified.
    // On other platforms it may be Discrete.
    // Both are valid — just ensure consistency with zero_copy.
    if p.memory == MemoryArchitecture::Unified {
        assert!(p.memory.supports_zero_copy());
    } else {
        assert!(!p.memory.supports_zero_copy());
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 17. Helper function consistency
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn helper_align_to_various_alignments() {
    // 64-byte alignment
    assert_eq!(align_to(1, 64), 64);
    assert_eq!(align_to(64, 64), 64);
    assert_eq!(align_to(65, 64), 128);

    // 4096-byte alignment (page)
    assert_eq!(align_to(1, 4096), 4096);
    assert_eq!(align_to(4096, 4096), 4096);
    assert_eq!(align_to(4097, 4096), 8192);
}

#[test]
fn helper_compute_threadgroup_size_basic() {
    let (gs, ng) = compute_threadgroup_size(1000, 256);
    assert_eq!(gs, 256);
    assert_eq!(ng, 4); // ceil(1000/256)
    assert!(gs * ng >= 1000);
}

#[test]
fn helper_compute_threadgroup_size_small_total() {
    let (gs, ng) = compute_threadgroup_size(10, 256);
    assert_eq!(gs, 10);
    assert_eq!(ng, 1);
}

#[test]
fn helper_compute_dispatch_groups_exact() {
    assert_eq!(compute_dispatch_groups(256, 256), 1);
    assert_eq!(compute_dispatch_groups(512, 256), 2);
}

#[test]
fn helper_compute_dispatch_groups_remainder() {
    assert_eq!(compute_dispatch_groups(257, 256), 2);
    assert_eq!(compute_dispatch_groups(1, 256), 1);
}
