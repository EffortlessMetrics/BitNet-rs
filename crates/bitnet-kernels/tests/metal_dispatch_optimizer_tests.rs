#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal dispatch optimizer tests for Apple Silicon GPU backend.
//!
//! Tests validate optimal workgroup sizing, thread occupancy,
//! and dispatch configuration for Metal compute shaders.

#![cfg(target_os = "macos")]

/// Metal GPU capabilities for dispatch planning.
#[derive(Debug, Clone)]
struct MetalGpuCapabilities {
    max_threads_per_threadgroup: u32,
    max_threadgroup_memory: u32,
    simd_width: u32,
    max_buffer_length: u64,
    unified_memory: bool,
}

impl MetalGpuCapabilities {
    fn apple_m1() -> Self {
        Self {
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory: 32768,
            simd_width: 32,
            max_buffer_length: 1 << 32, // 4 GiB
            unified_memory: true,
        }
    }

    fn apple_m2() -> Self {
        Self {
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory: 32768,
            simd_width: 32,
            max_buffer_length: 1 << 33, // 8 GiB
            unified_memory: true,
        }
    }

    fn apple_m3() -> Self {
        Self {
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory: 65536,
            simd_width: 32,
            max_buffer_length: 1 << 34, // 16 GiB
            unified_memory: true,
        }
    }
}

/// Dispatch configuration for a Metal compute pass.
#[derive(Debug, Clone, PartialEq)]
struct DispatchConfig {
    threadgroup_size: (u32, u32, u32),
    grid_size: (u32, u32, u32),
    threadgroup_memory: u32,
}

impl DispatchConfig {
    fn total_threads_per_group(&self) -> u32 {
        self.threadgroup_size.0 * self.threadgroup_size.1 * self.threadgroup_size.2
    }

    fn total_grid_groups(&self) -> u32 {
        self.grid_size.0 * self.grid_size.1 * self.grid_size.2
    }
}

/// Round `n` up to the next multiple of `align` (saturating).
fn round_up(n: u32, align: u32) -> u32 {
    assert!(align > 0);
    match n.checked_add(align - 1) {
        Some(v) => (v / align) * align,
        None => (n / align) * align, // already past max; floor-align
    }
}

/// Compute optimal 1-D dispatch for element-wise / vector kernels.
fn optimize_1d_dispatch(total_work: u32, caps: &MetalGpuCapabilities) -> DispatchConfig {
    // Pick threadgroup width as the largest SIMD-aligned value ≤ max_threads
    let tg_width = if total_work < caps.simd_width {
        caps.simd_width // minimum one SIMD wave
    } else {
        // Largest multiple of simd_width that fits in max_threads
        let max_aligned = (caps.max_threads_per_threadgroup / caps.simd_width) * caps.simd_width;
        // But do not exceed work size (capped to max_aligned)
        max_aligned.min(round_up(total_work, caps.simd_width))
    };

    let grid_x = total_work.saturating_add(tg_width - 1) / tg_width;

    DispatchConfig {
        threadgroup_size: (tg_width, 1, 1),
        grid_size: (grid_x, 1, 1),
        threadgroup_memory: 0,
    }
}

/// Compute optimal 2-D dispatch for image / feature-map kernels.
fn optimize_2d_dispatch(width: u32, height: u32, caps: &MetalGpuCapabilities) -> DispatchConfig {
    // Tile: prefer square-ish tiles aligned to SIMD width
    let tg_x = caps.simd_width; // 32 – one full SIMD lane per row
    let max_y = caps.max_threads_per_threadgroup / tg_x;
    // Cap Y to actual height to avoid waste
    let tg_y = max_y.min(height).max(1);

    let grid_x = (width + tg_x - 1) / tg_x;
    let grid_y = (height + tg_y - 1) / tg_y;

    DispatchConfig {
        threadgroup_size: (tg_x, tg_y, 1),
        grid_size: (grid_x, grid_y, 1),
        threadgroup_memory: 0,
    }
}

/// Compute optimal dispatch for tree-style reduction kernels.
///
/// The first pass reduces `input_size` elements inside each threadgroup
/// using shared memory. A second host-side pass (not modelled here)
/// reduces the per-group partial results.
fn optimize_reduction_dispatch(input_size: u32, caps: &MetalGpuCapabilities) -> DispatchConfig {
    let simd = caps.simd_width;
    // Threads per group: SIMD-aligned, ≤ max, ≤ input_size (rounded up)
    let threads = {
        let aligned = round_up(input_size, simd).min(caps.max_threads_per_threadgroup);
        // Make sure it's a multiple of simd_width
        (aligned / simd) * simd
    }
    .max(simd); // at least one wave

    // Shared memory: one f32 per thread for partial sums
    let smem = threads * 4; // sizeof(f32) == 4
    let smem = smem.min(caps.max_threadgroup_memory);

    let groups = (input_size + threads - 1) / threads;

    DispatchConfig {
        threadgroup_size: (threads, 1, 1),
        grid_size: (groups, 1, 1),
        threadgroup_memory: smem,
    }
}

/// Compute optimal dispatch for matrix-multiply (M×K) * (K×N) → (M×N).
///
/// Uses a 2-D tile strategy: each threadgroup computes a tile of the
/// output matrix.  Tile dimensions are SIMD-aligned.
fn optimize_matmul_dispatch(
    m: u32,
    n: u32,
    _k: u32,
    caps: &MetalGpuCapabilities,
) -> DispatchConfig {
    let simd = caps.simd_width;

    // Tile: 32×32 is the sweet spot for Apple Silicon (one SIMD per row)
    let tile_m = simd; // 32
    let tile_n = caps.max_threads_per_threadgroup / tile_m; // 1024/32 = 32
    let tile_n = tile_n.min(simd); // cap to 32

    let tg_x = tile_n;
    let tg_y = tile_m;

    // Shared memory for two input tiles: A_tile (tile_m × K_TILE) + B_tile (K_TILE × tile_n)
    // Use K_TILE = 8 as a conservative tile depth
    let k_tile: u32 = 8;
    let smem = (tile_m * k_tile + k_tile * tile_n) * 4; // f32
    let smem = smem.min(caps.max_threadgroup_memory);

    let grid_x = (n + tile_n - 1) / tile_n;
    let grid_y = (m + tile_m - 1) / tile_m;

    DispatchConfig {
        threadgroup_size: (tg_x, tg_y, 1),
        grid_size: (grid_x, grid_y, 1),
        threadgroup_memory: smem,
    }
}

/// Estimate thread occupancy (0.0 – 1.0) for a dispatch configuration.
///
/// A simple model: occupancy = (threads per group / max threads) clamped to [0, 1].
fn compute_occupancy(config: &DispatchConfig, caps: &MetalGpuCapabilities) -> f32 {
    let threads = config.total_threads_per_group() as f32;
    let max = caps.max_threads_per_threadgroup as f32;
    (threads / max).min(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===== GPU capability profiles =====

    #[test]
    fn m1_capabilities_are_correct() {
        let caps = MetalGpuCapabilities::apple_m1();
        assert_eq!(caps.max_threads_per_threadgroup, 1024);
        assert_eq!(caps.max_threadgroup_memory, 32768);
        assert_eq!(caps.simd_width, 32);
        assert!(caps.unified_memory);
    }

    #[test]
    fn m2_capabilities_are_correct() {
        let caps = MetalGpuCapabilities::apple_m2();
        assert_eq!(caps.max_threads_per_threadgroup, 1024);
        assert_eq!(caps.max_buffer_length, 1 << 33);
        assert!(caps.unified_memory);
    }

    #[test]
    fn m3_capabilities_have_larger_threadgroup_memory() {
        let caps = MetalGpuCapabilities::apple_m3();
        assert_eq!(caps.max_threadgroup_memory, 65536);
        assert_eq!(caps.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn all_profiles_have_simd_width_32() {
        for caps in [
            MetalGpuCapabilities::apple_m1(),
            MetalGpuCapabilities::apple_m2(),
            MetalGpuCapabilities::apple_m3(),
        ] {
            assert_eq!(caps.simd_width, 32, "Apple Silicon SIMD width must be 32");
        }
    }

    #[test]
    fn all_profiles_are_unified_memory() {
        for caps in [
            MetalGpuCapabilities::apple_m1(),
            MetalGpuCapabilities::apple_m2(),
            MetalGpuCapabilities::apple_m3(),
        ] {
            assert!(caps.unified_memory);
        }
    }

    // ===== 1-D dispatch =====

    #[test]
    fn dispatch_1d_small_work() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(16, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32, "min one SIMD wave");
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn dispatch_1d_exact_simd() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(32, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn dispatch_1d_256_elements() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(256, &caps);
        assert_eq!(cfg.threadgroup_size.0 % 32, 0);
        let total_coverage = cfg.threadgroup_size.0 * cfg.grid_size.0;
        assert!(total_coverage >= 256);
    }

    #[test]
    fn dispatch_1d_1024_exact_fit() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(1024, &caps);
        assert_eq!(cfg.threadgroup_size.0, 1024);
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn dispatch_1d_large_work_multiple_groups() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(4096, &caps);
        assert_eq!(cfg.threadgroup_size.0, 1024);
        assert_eq!(cfg.grid_size.0, 4);
    }

    #[test]
    fn dispatch_1d_non_power_of_two() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(1000, &caps);
        assert_eq!(cfg.threadgroup_size.0 % 32, 0, "SIMD aligned");
        let total_coverage = cfg.threadgroup_size.0 * cfg.grid_size.0;
        assert!(total_coverage >= 1000);
    }

    #[test]
    fn dispatch_1d_very_large() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(1_000_000, &caps);
        assert_eq!(cfg.threadgroup_size.0, 1024);
        assert!(cfg.grid_size.0 >= 977); // ceil(1M / 1024)
    }

    #[test]
    fn dispatch_1d_one_element() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(1, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32, "min wave");
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn dispatch_1d_simd_aligned_threadgroup() {
        let caps = MetalGpuCapabilities::apple_m1();
        for work in [33, 63, 65, 127, 129, 255, 513, 1025] {
            let cfg = optimize_1d_dispatch(work, &caps);
            assert_eq!(
                cfg.threadgroup_size.0 % caps.simd_width,
                0,
                "work={work}: threadgroup must be SIMD-aligned"
            );
        }
    }

    #[test]
    fn dispatch_1d_covers_all_work() {
        let caps = MetalGpuCapabilities::apple_m1();
        for work in [1, 31, 32, 33, 64, 100, 1023, 1024, 1025, 10000] {
            let cfg = optimize_1d_dispatch(work, &caps);
            let total = cfg.threadgroup_size.0 * cfg.grid_size.0;
            assert!(total >= work, "work={work}: dispatched {total} threads < {work}");
        }
    }

    #[test]
    fn dispatch_1d_y_z_always_one() {
        let caps = MetalGpuCapabilities::apple_m2();
        let cfg = optimize_1d_dispatch(2048, &caps);
        assert_eq!(cfg.threadgroup_size.1, 1);
        assert_eq!(cfg.threadgroup_size.2, 1);
        assert_eq!(cfg.grid_size.1, 1);
        assert_eq!(cfg.grid_size.2, 1);
    }

    #[test]
    fn dispatch_1d_no_shared_memory() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(512, &caps);
        assert_eq!(cfg.threadgroup_memory, 0);
    }

    // ===== 2-D dispatch =====

    #[test]
    fn dispatch_2d_small_image() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(8, 8, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        assert!(cfg.threadgroup_size.1 >= 1);
    }

    #[test]
    fn dispatch_2d_square_256() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(256, 256, &caps);
        let tg_total = cfg.total_threads_per_group();
        assert!(tg_total <= caps.max_threads_per_threadgroup);
        assert_eq!(cfg.threadgroup_size.0 % 32, 0);
    }

    #[test]
    fn dispatch_2d_wide_image() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(4096, 16, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        assert!(cfg.threadgroup_size.1 <= 16);
        let covered_x = cfg.threadgroup_size.0 * cfg.grid_size.0;
        assert!(covered_x >= 4096);
    }

    #[test]
    fn dispatch_2d_tall_image() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(16, 4096, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        let covered_y = cfg.threadgroup_size.1 * cfg.grid_size.1;
        assert!(covered_y >= 4096);
    }

    #[test]
    fn dispatch_2d_covers_work() {
        let caps = MetalGpuCapabilities::apple_m1();
        for (w, h) in [(1, 1), (7, 13), (32, 32), (100, 200), (1024, 1024)] {
            let cfg = optimize_2d_dispatch(w, h, &caps);
            let cx = cfg.threadgroup_size.0 * cfg.grid_size.0;
            let cy = cfg.threadgroup_size.1 * cfg.grid_size.1;
            assert!(cx >= w, "w={w},h={h}: x coverage {cx} < {w}");
            assert!(cy >= h, "w={w},h={h}: y coverage {cy} < {h}");
        }
    }

    #[test]
    fn dispatch_2d_respects_max_threads() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(2048, 2048, &caps);
        assert!(cfg.total_threads_per_group() <= caps.max_threads_per_threadgroup);
    }

    #[test]
    fn dispatch_2d_z_always_one() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(512, 512, &caps);
        assert_eq!(cfg.threadgroup_size.2, 1);
        assert_eq!(cfg.grid_size.2, 1);
    }

    #[test]
    fn dispatch_2d_no_shared_memory() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(128, 128, &caps);
        assert_eq!(cfg.threadgroup_memory, 0);
    }

    #[test]
    fn dispatch_2d_single_pixel() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(1, 1, &caps);
        assert!(cfg.total_threads_per_group() >= 1);
        assert_eq!(cfg.grid_size.0, 1);
        assert_eq!(cfg.grid_size.1, 1);
    }

    // ===== Reduction dispatch =====

    #[test]
    fn reduction_small_input() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(16, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32, "at least one SIMD wave");
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn reduction_exact_simd() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(32, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn reduction_shared_memory_allocated() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(1024, &caps);
        assert!(cfg.threadgroup_memory > 0, "reduction needs shared memory");
        assert_eq!(cfg.threadgroup_memory, cfg.threadgroup_size.0 * 4);
    }

    #[test]
    fn reduction_shared_memory_within_budget() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(100_000, &caps);
        assert!(cfg.threadgroup_memory <= caps.max_threadgroup_memory);
    }

    #[test]
    fn reduction_large_input_multi_group() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(100_000, &caps);
        assert!(cfg.grid_size.0 > 1, "large reductions need >1 group");
        let total_coverage = cfg.threadgroup_size.0 * cfg.grid_size.0;
        assert!(total_coverage >= 100_000);
    }

    #[test]
    fn reduction_simd_aligned() {
        let caps = MetalGpuCapabilities::apple_m1();
        for size in [33, 100, 500, 1023, 2048, 65537] {
            let cfg = optimize_reduction_dispatch(size, &caps);
            assert_eq!(
                cfg.threadgroup_size.0 % caps.simd_width,
                0,
                "size={size}: threadgroup not SIMD-aligned"
            );
        }
    }

    #[test]
    fn reduction_one_element() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(1, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn reduction_m3_allows_more_shared_memory() {
        let m1 = MetalGpuCapabilities::apple_m1();
        let m3 = MetalGpuCapabilities::apple_m3();
        let cfg_m1 = optimize_reduction_dispatch(1024, &m1);
        let cfg_m3 = optimize_reduction_dispatch(1024, &m3);
        // Same dispatch since 1024 fits in both, but budget differs
        assert!(m3.max_threadgroup_memory >= m1.max_threadgroup_memory);
        assert!(cfg_m3.threadgroup_memory <= m3.max_threadgroup_memory);
        assert!(cfg_m1.threadgroup_memory <= m1.max_threadgroup_memory);
    }

    // ===== Matmul dispatch =====

    #[test]
    fn matmul_small_square() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(32, 32, 32, &caps);
        assert_eq!(cfg.grid_size.0, 1);
        assert_eq!(cfg.grid_size.1, 1);
    }

    #[test]
    fn matmul_threadgroup_32x32() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(256, 256, 256, &caps);
        assert_eq!(cfg.threadgroup_size.0, 32);
        assert_eq!(cfg.threadgroup_size.1, 32);
        assert_eq!(cfg.total_threads_per_group(), 1024);
    }

    #[test]
    fn matmul_uses_shared_memory() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(128, 128, 128, &caps);
        assert!(cfg.threadgroup_memory > 0, "matmul needs shared memory for tiles");
    }

    #[test]
    fn matmul_shared_memory_within_budget() {
        for caps in [
            MetalGpuCapabilities::apple_m1(),
            MetalGpuCapabilities::apple_m2(),
            MetalGpuCapabilities::apple_m3(),
        ] {
            let cfg = optimize_matmul_dispatch(4096, 4096, 4096, &caps);
            assert!(
                cfg.threadgroup_memory <= caps.max_threadgroup_memory,
                "smem {} > budget {}",
                cfg.threadgroup_memory,
                caps.max_threadgroup_memory
            );
        }
    }

    #[test]
    fn matmul_grid_covers_output() {
        let caps = MetalGpuCapabilities::apple_m1();
        let (m, n, k) = (1000, 2000, 512);
        let cfg = optimize_matmul_dispatch(m, n, k, &caps);
        let covered_m = cfg.threadgroup_size.1 * cfg.grid_size.1;
        let covered_n = cfg.threadgroup_size.0 * cfg.grid_size.0;
        assert!(covered_m >= m, "M coverage {covered_m} < {m}");
        assert!(covered_n >= n, "N coverage {covered_n} < {n}");
    }

    #[test]
    fn matmul_non_square() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(64, 2048, 256, &caps);
        assert!(cfg.grid_size.0 > cfg.grid_size.1, "wider N → more X groups");
    }

    #[test]
    fn matmul_respects_max_threads() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(4096, 4096, 4096, &caps);
        assert!(cfg.total_threads_per_group() <= caps.max_threads_per_threadgroup);
    }

    #[test]
    fn matmul_tiny_dimensions() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(1, 1, 1, &caps);
        assert!(cfg.total_threads_per_group() >= 1);
        assert_eq!(cfg.grid_size.0, 1);
        assert_eq!(cfg.grid_size.1, 1);
    }

    #[test]
    fn matmul_k_independent_of_grid() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg_a = optimize_matmul_dispatch(256, 256, 64, &caps);
        let cfg_b = optimize_matmul_dispatch(256, 256, 4096, &caps);
        assert_eq!(cfg_a.grid_size, cfg_b.grid_size, "K does not affect grid");
        assert_eq!(cfg_a.threadgroup_size, cfg_b.threadgroup_size);
    }

    // ===== Occupancy =====

    #[test]
    fn occupancy_full_threadgroup() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(1024, &caps);
        let occ = compute_occupancy(&cfg, &caps);
        assert!((occ - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn occupancy_single_wave() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(1, &caps);
        let occ = compute_occupancy(&cfg, &caps);
        let expected = 32.0 / 1024.0;
        assert!((occ - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn occupancy_half() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(512, &caps);
        let occ = compute_occupancy(&cfg, &caps);
        assert!((occ - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn occupancy_never_exceeds_one() {
        let caps = MetalGpuCapabilities::apple_m1();
        for work in [1, 32, 100, 1024, 5000, 1_000_000] {
            let cfg = optimize_1d_dispatch(work, &caps);
            let occ = compute_occupancy(&cfg, &caps);
            assert!(occ <= 1.0, "work={work}: occupancy {occ} > 1.0");
        }
    }

    #[test]
    fn occupancy_matmul_full() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(256, 256, 256, &caps);
        let occ = compute_occupancy(&cfg, &caps);
        assert!((occ - 1.0).abs() < f32::EPSILON, "32×32 tile = 1024 = full");
    }

    #[test]
    fn occupancy_2d_within_bounds() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(64, 64, &caps);
        let occ = compute_occupancy(&cfg, &caps);
        assert!(occ > 0.0 && occ <= 1.0);
    }

    #[test]
    fn occupancy_reduction_positive() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_reduction_dispatch(256, &caps);
        let occ = compute_occupancy(&cfg, &caps);
        assert!(occ > 0.0);
    }

    // ===== Cross-profile consistency =====

    #[test]
    fn same_dispatch_across_m1_m2_for_small_work() {
        let m1 = MetalGpuCapabilities::apple_m1();
        let m2 = MetalGpuCapabilities::apple_m2();
        let cfg1 = optimize_1d_dispatch(256, &m1);
        let cfg2 = optimize_1d_dispatch(256, &m2);
        assert_eq!(cfg1, cfg2, "M1 and M2 share thread limits");
    }

    #[test]
    fn m3_threadgroup_memory_larger_than_m1() {
        let m1 = MetalGpuCapabilities::apple_m1();
        let m3 = MetalGpuCapabilities::apple_m3();
        assert!(m3.max_threadgroup_memory > m1.max_threadgroup_memory);
    }

    #[test]
    fn dispatch_deterministic() {
        let caps = MetalGpuCapabilities::apple_m1();
        let a = optimize_1d_dispatch(777, &caps);
        let b = optimize_1d_dispatch(777, &caps);
        assert_eq!(a, b, "same input → same dispatch");
    }

    // ===== Helper function tests =====

    #[test]
    fn round_up_exact() {
        assert_eq!(round_up(32, 32), 32);
    }

    #[test]
    fn round_up_non_aligned() {
        assert_eq!(round_up(33, 32), 64);
    }

    #[test]
    fn round_up_one() {
        assert_eq!(round_up(1, 32), 32);
    }

    #[test]
    fn round_up_zero() {
        assert_eq!(round_up(0, 32), 0);
    }

    #[test]
    fn total_threads_per_group_1d() {
        let cfg = DispatchConfig {
            threadgroup_size: (256, 1, 1),
            grid_size: (4, 1, 1),
            threadgroup_memory: 0,
        };
        assert_eq!(cfg.total_threads_per_group(), 256);
    }

    #[test]
    fn total_threads_per_group_2d() {
        let cfg = DispatchConfig {
            threadgroup_size: (32, 32, 1),
            grid_size: (1, 1, 1),
            threadgroup_memory: 0,
        };
        assert_eq!(cfg.total_threads_per_group(), 1024);
    }

    #[test]
    fn total_grid_groups_multi() {
        let cfg = DispatchConfig {
            threadgroup_size: (32, 1, 1),
            grid_size: (4, 8, 1),
            threadgroup_memory: 0,
        };
        assert_eq!(cfg.total_grid_groups(), 32);
    }

    // ===== Edge / stress =====

    #[test]
    fn dispatch_1d_max_u32_width_does_not_panic() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_1d_dispatch(u32::MAX, &caps);
        assert_eq!(cfg.threadgroup_size.0, 1024);
        assert!(cfg.grid_size.0 > 0);
    }

    #[test]
    fn dispatch_2d_height_one() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(1024, 1, &caps);
        assert_eq!(cfg.threadgroup_size.1, 1);
        assert_eq!(cfg.grid_size.1, 1);
    }

    #[test]
    fn dispatch_2d_width_one() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_2d_dispatch(1, 1024, &caps);
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn reduction_power_of_two_sizes() {
        let caps = MetalGpuCapabilities::apple_m1();
        for exp in 0..20 {
            let sz = 1u32 << exp;
            let cfg = optimize_reduction_dispatch(sz, &caps);
            assert_eq!(cfg.threadgroup_size.0 % 32, 0);
            assert!(cfg.threadgroup_memory <= caps.max_threadgroup_memory);
        }
    }

    #[test]
    fn matmul_large_dimensions_no_overflow() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cfg = optimize_matmul_dispatch(16384, 16384, 16384, &caps);
        assert!(cfg.grid_size.0 > 0);
        assert!(cfg.grid_size.1 > 0);
    }

    #[test]
    fn dispatch_config_debug_format() {
        let cfg = DispatchConfig {
            threadgroup_size: (32, 1, 1),
            grid_size: (1, 1, 1),
            threadgroup_memory: 0,
        };
        let s = format!("{cfg:?}");
        assert!(s.contains("threadgroup_size"));
        assert!(s.contains("grid_size"));
    }

    #[test]
    fn dispatch_config_clone_eq() {
        let cfg = DispatchConfig {
            threadgroup_size: (32, 32, 1),
            grid_size: (8, 8, 1),
            threadgroup_memory: 2048,
        };
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    #[test]
    fn gpu_capabilities_clone() {
        let caps = MetalGpuCapabilities::apple_m1();
        let cloned = caps.clone();
        assert_eq!(caps.simd_width, cloned.simd_width);
        assert_eq!(caps.max_threads_per_threadgroup, cloned.max_threads_per_threadgroup);
    }
}
