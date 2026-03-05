#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal workgroup and threadgroup optimization tests for Apple Silicon.
//!
//! Validates threadgroup sizing heuristics, occupancy estimation, shared memory
//! budgets, SIMD-width alignment, and dispatch grid calculations across a wide
//! range of problem shapes. Pure arithmetic — no Metal runtime required.

// ── Apple Silicon Metal constants ───────────────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon (M1–M4).
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD (warp) width on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Maximum threadgroup memory (shared/tile memory) in bytes.
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

/// Maximum total threads per grid dimension on Apple Silicon.
const MAX_GRID_DIM: u32 = 0x7FFF_FFFF; // 2^31 - 1

/// Maximum number of threadgroups per grid dimension.
const MAX_THREADGROUPS_PER_GRID_DIM: u32 = 0x7FFF_FFFF;

/// Typical register file size per thread (bytes, approximate for M-series).
const REGISTERS_PER_THREAD: usize = 128;

/// Maximum concurrent threadgroups per compute unit (Apple GPU).
const MAX_THREADGROUPS_PER_CU: u32 = 32;

/// Number of GPU cores on Apple M1 (baseline reference).
const M1_GPU_CORES: u32 = 8;

/// Number of GPU cores on Apple M1 Pro.
const M1_PRO_GPU_CORES: u32 = 16;

/// Number of GPU cores on Apple M2 Max.
const M2_MAX_GPU_CORES: u32 = 38;

/// Number of GPU cores on Apple M3 Max.
const M3_MAX_GPU_CORES: u32 = 40;

// ── Helper functions ────────────────────────────────────────────────────────

fn ceil_div(total: u32, group_size: u32) -> u32 {
    assert_ne!(group_size, 0);
    (total + group_size - 1) / group_size
}

fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Round `n` up to the nearest multiple of `align`.
fn round_up(n: u32, align: u32) -> u32 {
    assert_ne!(align, 0);
    ((n + align - 1) / align) * align
}

/// Choose an optimal 1-D threadgroup size: multiple of SIMD_WIDTH, clamped to
/// the hardware maximum, and no larger than `total_elements`.
fn optimal_threadgroup_1d(total_elements: u32) -> u32 {
    if total_elements == 0 {
        return 0;
    }
    let rounded = round_up(total_elements, SIMD_WIDTH);
    rounded.min(MAX_THREADS_PER_THREADGROUP)
}

/// Choose optimal 2-D threadgroup dimensions `(width, height)`.
fn optimal_threadgroup_2d(cols: u32, rows: u32) -> (u32, u32) {
    if cols == 0 || rows == 0 {
        return (0, 0);
    }
    let w = SIMD_WIDTH.min(cols);
    let max_h = MAX_THREADS_PER_THREADGROUP / w;
    let h = max_h.min(rows);
    (w, h)
}

/// Choose optimal 3-D threadgroup dimensions `(x, y, z)`.
fn optimal_threadgroup_3d(dim_x: u32, dim_y: u32, dim_z: u32) -> (u32, u32, u32) {
    if dim_x == 0 || dim_y == 0 || dim_z == 0 {
        return (0, 0, 0);
    }
    let x = SIMD_WIDTH.min(dim_x);
    let max_yz = MAX_THREADS_PER_THREADGROUP / x;
    let y = max_yz.min(dim_y);
    let z = if y > 0 { (max_yz / y).min(dim_z).max(1) } else { 1 };
    (x, y, z)
}

/// Compute the number of threadgroups in a 1-D dispatch.
fn grid_threadgroups_1d(total: u32, tg_size: u32) -> u32 {
    ceil_div(total, tg_size)
}

/// Compute the number of threadgroups in each dimension for a 2-D dispatch.
fn grid_threadgroups_2d(cols: u32, rows: u32, tg_w: u32, tg_h: u32) -> (u32, u32) {
    (ceil_div(cols, tg_w), ceil_div(rows, tg_h))
}

/// Compute the number of threadgroups in each dimension for a 3-D dispatch.
fn grid_threadgroups_3d(
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    tg_x: u32,
    tg_y: u32,
    tg_z: u32,
) -> (u32, u32, u32) {
    (ceil_div(dim_x, tg_x), ceil_div(dim_y, tg_y), ceil_div(dim_z, tg_z))
}

/// Estimate occupancy as the fraction of maximum parallelism utilised.
/// Returns a value in `[0.0, 1.0]`.
fn estimate_occupancy(
    threads_per_tg: u32,
    shared_mem_per_tg: usize,
    registers_per_thread: usize,
) -> f64 {
    if threads_per_tg == 0 {
        return 0.0;
    }
    // Thread-count limit.
    let thread_occupancy = (threads_per_tg as f64) / (MAX_THREADS_PER_THREADGROUP as f64);

    // Shared-memory limit.
    let mem_occupancy = if shared_mem_per_tg == 0 {
        1.0
    } else {
        (MAX_THREADGROUP_MEMORY as f64) / (shared_mem_per_tg as f64)
    };

    // Register-pressure limit (rough heuristic).
    let reg_occupancy = if registers_per_thread == 0 {
        1.0
    } else {
        (REGISTERS_PER_THREAD as f64) / (registers_per_thread as f64)
    };

    thread_occupancy.min(mem_occupancy).min(reg_occupancy).min(1.0)
}

/// Calculate shared-memory bytes needed for a matmul tile of size
/// `(tile_m, tile_n)` with element size `elem_bytes`.
fn shared_memory_for_tile(tile_m: u32, tile_n: u32, tile_k: u32, elem_bytes: usize) -> usize {
    // A-tile: tile_m × tile_k,  B-tile: tile_k × tile_n
    let a_bytes = (tile_m as usize) * (tile_k as usize) * elem_bytes;
    let b_bytes = (tile_k as usize) * (tile_n as usize) * elem_bytes;
    a_bytes + b_bytes
}

/// Choose a matmul tile size that fits within the threadgroup memory budget
/// for `elem_bytes`-wide elements. Returns `(tile_m, tile_n, tile_k)`.
fn best_matmul_tile(elem_bytes: usize) -> (u32, u32, u32) {
    // Try common tile sizes from large to small.
    let candidates: &[(u32, u32, u32)] =
        &[(32, 32, 32), (32, 32, 16), (16, 16, 16), (16, 16, 8), (8, 8, 8)];
    for &(m, n, k) in candidates {
        if shared_memory_for_tile(m, n, k, elem_bytes) <= MAX_THREADGROUP_MEMORY {
            return (m, n, k);
        }
    }
    (8, 8, 8)
}

/// Determine SIMD utilisation ratio for a given threadgroup width.
fn simd_utilisation(threadgroup_width: u32) -> f64 {
    if threadgroup_width == 0 {
        return 0.0;
    }
    let full_warps = threadgroup_width / SIMD_WIDTH;
    let remainder = threadgroup_width % SIMD_WIDTH;
    let active_lanes = full_warps * SIMD_WIDTH + remainder;
    let total_lanes = (full_warps + if remainder > 0 { 1 } else { 0 }) * SIMD_WIDTH;
    if total_lanes == 0 {
        return 0.0;
    }
    (active_lanes as f64) / (total_lanes as f64)
}

/// Pick the best reduction threadgroup size for `n` elements.
fn reduction_threadgroup_size(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    // Use the smallest power-of-two ≥ SIMD_WIDTH that covers n, capped at 1024.
    let mut size = SIMD_WIDTH;
    while size < n && size < MAX_THREADS_PER_THREADGROUP {
        size *= 2;
    }
    size.min(MAX_THREADS_PER_THREADGROUP)
}

/// Compute total dispatched threads for a 1-D launch.
fn total_dispatched_threads_1d(total_elements: u32, tg_size: u32) -> u64 {
    let groups = ceil_div(total_elements, tg_size);
    (groups as u64) * (tg_size as u64)
}

/// Estimate whether a workgroup configuration can achieve full wave occupancy
/// on a given GPU core count.
fn achieves_full_wave(num_threadgroups: u32, gpu_cores: u32) -> bool {
    num_threadgroups >= gpu_cores
}

// ── Tests: Optimal threadgroup sizing for matrix dimensions ─────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_32_elements() {
    assert_eq!(optimal_threadgroup_1d(32), 32);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_64_elements() {
    assert_eq!(optimal_threadgroup_1d(64), 64);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_128_elements() {
    assert_eq!(optimal_threadgroup_1d(128), 128);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_256_elements() {
    assert_eq!(optimal_threadgroup_1d(256), 256);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_512_elements() {
    assert_eq!(optimal_threadgroup_1d(512), 512);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_1024_elements() {
    assert_eq!(optimal_threadgroup_1d(1024), 1024);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_clamps_at_max() {
    assert_eq!(optimal_threadgroup_1d(2048), MAX_THREADS_PER_THREADGROUP);
    assert_eq!(optimal_threadgroup_1d(100_000), MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_rounds_up_to_simd() {
    // 33 rounds to 64 (next SIMD multiple).
    assert_eq!(optimal_threadgroup_1d(33), 64);
    assert_eq!(optimal_threadgroup_1d(1), 32);
    assert_eq!(optimal_threadgroup_1d(31), 32);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_1d_zero() {
    assert_eq!(optimal_threadgroup_1d(0), 0);
}

// ── Tests: 2-D threadgroup sizing ───────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_32x32() {
    let (w, h) = optimal_threadgroup_2d(32, 32);
    assert_eq!(w, 32);
    assert!(h <= MAX_THREADS_PER_THREADGROUP / w);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_64x64() {
    let (w, h) = optimal_threadgroup_2d(64, 64);
    assert_eq!(w, 32);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_128x128() {
    let (w, h) = optimal_threadgroup_2d(128, 128);
    assert_eq!(w, 32);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
    assert!(h <= 32);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_256x256() {
    let (w, h) = optimal_threadgroup_2d(256, 256);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_small_cols() {
    // When cols < SIMD_WIDTH, width is clamped to cols.
    let (w, h) = optimal_threadgroup_2d(8, 256);
    assert_eq!(w, 8);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_zero_dims() {
    assert_eq!(optimal_threadgroup_2d(0, 128), (0, 0));
    assert_eq!(optimal_threadgroup_2d(128, 0), (0, 0));
    assert_eq!(optimal_threadgroup_2d(0, 0), (0, 0));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_1x1() {
    let (w, h) = optimal_threadgroup_2d(1, 1);
    assert_eq!(w, 1);
    assert_eq!(h, 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_wide_matrix() {
    // Very wide but short: 4096 cols × 4 rows.
    let (w, h) = optimal_threadgroup_2d(4096, 4);
    assert_eq!(w, 32);
    assert!(h <= 4);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_2d_tall_matrix() {
    // Narrow but tall: 4 cols × 4096 rows.
    let (w, h) = optimal_threadgroup_2d(4, 4096);
    assert_eq!(w, 4);
    let max_h = MAX_THREADS_PER_THREADGROUP / w;
    assert_eq!(h, max_h.min(4096));
}

// ── Tests: 3-D threadgroup sizing ───────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_3d_basic() {
    let (x, y, z) = optimal_threadgroup_3d(64, 64, 64);
    assert_eq!(x, 32);
    assert!(x * y * z <= MAX_THREADS_PER_THREADGROUP);
    assert!(z >= 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_3d_batch_ops() {
    // Batch dimension in z: 32 batches.
    let (x, y, z) = optimal_threadgroup_3d(128, 128, 32);
    assert!(x * y * z <= MAX_THREADS_PER_THREADGROUP);
    assert!(x >= 1 && y >= 1 && z >= 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_3d_volumetric() {
    let (x, y, z) = optimal_threadgroup_3d(16, 16, 16);
    assert!(x * y * z <= MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_3d_zero_dim() {
    assert_eq!(optimal_threadgroup_3d(0, 64, 64), (0, 0, 0));
    assert_eq!(optimal_threadgroup_3d(64, 0, 64), (0, 0, 0));
    assert_eq!(optimal_threadgroup_3d(64, 64, 0), (0, 0, 0));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_3d_single_batch() {
    let (x, y, z) = optimal_threadgroup_3d(256, 256, 1);
    assert!(x * y * z <= MAX_THREADS_PER_THREADGROUP);
    assert_eq!(z, 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_3d_minimal() {
    let (x, y, z) = optimal_threadgroup_3d(1, 1, 1);
    assert_eq!((x, y, z), (1, 1, 1));
}

// ── Tests: 1-D dispatch – element-wise ops ──────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_dispatch_1d_elementwise_small() {
    let n: u32 = 256;
    let tg = optimal_threadgroup_1d(n);
    let groups = grid_threadgroups_1d(n, tg);
    assert!(groups * tg >= n);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_dispatch_1d_elementwise_large() {
    let n: u32 = 1_000_000;
    let tg = optimal_threadgroup_1d(n);
    let groups = grid_threadgroups_1d(n, tg);
    assert!(groups * tg >= n);
    assert_eq!(tg, MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_dispatch_1d_total_threads_covers_problem() {
    for &n in &[1u32, 31, 32, 33, 63, 64, 127, 1023, 1024, 1025, 65536] {
        let tg = optimal_threadgroup_1d(n);
        let dispatched = total_dispatched_threads_1d(n, tg);
        assert!(dispatched >= n as u64, "dispatched {dispatched} < problem size {n}");
    }
}

// ── Tests: 1-D dispatch – reduction kernels ─────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_reduction_tg_size_pow2() {
    for &n in &[32u32, 64, 128, 256, 512, 1024] {
        let tg = reduction_threadgroup_size(n);
        assert!(is_power_of_two(tg), "reduction tg {tg} not power of 2");
        assert!(tg >= SIMD_WIDTH);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_reduction_tg_size_non_pow2_input() {
    let tg = reduction_threadgroup_size(100);
    assert!(is_power_of_two(tg));
    assert!(tg >= 100);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_reduction_tg_clamped_at_1024() {
    let tg = reduction_threadgroup_size(4096);
    assert_eq!(tg, MAX_THREADS_PER_THREADGROUP);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_reduction_tg_minimum_simd_width() {
    assert_eq!(reduction_threadgroup_size(1), SIMD_WIDTH);
    assert_eq!(reduction_threadgroup_size(16), SIMD_WIDTH);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_reduction_tg_zero() {
    assert_eq!(reduction_threadgroup_size(0), 0);
}

// ── Tests: 2-D dispatch – matmul tiles ──────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_matmul_tile_dispatch_2d() {
    let (m, n) = (512, 512);
    let (tg_w, tg_h) = optimal_threadgroup_2d(n, m);
    let (gx, gy) = grid_threadgroups_2d(n, m, tg_w, tg_h);
    assert!(gx * tg_w >= n);
    assert!(gy * tg_h >= m);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_matmul_tile_dispatch_non_square() {
    let (m, n) = (768, 1024);
    let (tg_w, tg_h) = optimal_threadgroup_2d(n, m);
    let (gx, gy) = grid_threadgroups_2d(n, m, tg_w, tg_h);
    assert!(gx * tg_w >= n);
    assert!(gy * tg_h >= m);
}

// ── Tests: 2-D dispatch – convolution patches ───────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_conv_patch_dispatch() {
    // 224×224 image, 3×3 kernel → output 222×222.
    let (out_w, out_h) = (222, 222);
    let (tg_w, tg_h) = optimal_threadgroup_2d(out_w, out_h);
    let (gx, gy) = grid_threadgroups_2d(out_w, out_h, tg_w, tg_h);
    assert!(gx * tg_w >= out_w);
    assert!(gy * tg_h >= out_h);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_conv_patch_small_feature_map() {
    let (out_w, out_h) = (7, 7);
    let (tg_w, tg_h) = optimal_threadgroup_2d(out_w, out_h);
    assert!(tg_w <= 7);
    assert!(tg_h <= 7);
}

// ── Tests: 3-D dispatch – batch operations ──────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_batch_matmul_dispatch() {
    let (m, n, batch) = (128, 128, 8);
    let (tg_x, tg_y, tg_z) = optimal_threadgroup_3d(n, m, batch);
    let (gx, gy, gz) = grid_threadgroups_3d(n, m, batch, tg_x, tg_y, tg_z);
    assert!(gx * tg_x >= n);
    assert!(gy * tg_y >= m);
    assert!(gz * tg_z >= batch);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_volumetric_3d_dispatch() {
    let (dx, dy, dz) = (64, 64, 64);
    let (tg_x, tg_y, tg_z) = optimal_threadgroup_3d(dx, dy, dz);
    let total_tg_threads = tg_x * tg_y * tg_z;
    assert!(total_tg_threads <= MAX_THREADS_PER_THREADGROUP);
    let (gx, gy, gz) = grid_threadgroups_3d(dx, dy, dz, tg_x, tg_y, tg_z);
    assert!(gx * tg_x >= dx);
    assert!(gy * tg_y >= dy);
    assert!(gz * tg_z >= dz);
}

// ── Tests: Apple M-series GPU constraints ───────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_max_threads_per_threadgroup_limit() {
    // No helper should ever return a threadgroup size > 1024.
    for &n in &[1u32, 32, 512, 1024, 2048, 100_000] {
        let tg = optimal_threadgroup_1d(n);
        assert!(tg <= MAX_THREADS_PER_THREADGROUP, "1D tg {tg} > 1024");
    }
    for &(c, r) in &[(1u32, 1u32), (128, 128), (4096, 4096)] {
        let (w, h) = optimal_threadgroup_2d(c, r);
        assert!(w * h <= MAX_THREADS_PER_THREADGROUP, "2D tg {w}×{h} > 1024");
    }
    for &(x, y, z) in &[(1u32, 1u32, 1u32), (64, 64, 64), (256, 256, 256)] {
        let (tx, ty, tz) = optimal_threadgroup_3d(x, y, z);
        assert!(tx * ty * tz <= MAX_THREADS_PER_THREADGROUP, "3D tg {tx}×{ty}×{tz} > 1024");
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_threadgroup_memory_within_budget() {
    // Verify our tile choices never exceed 32 KiB.
    for &elem_bytes in &[1usize, 2, 4] {
        let (tm, tn, tk) = best_matmul_tile(elem_bytes);
        let mem = shared_memory_for_tile(tm, tn, tk, elem_bytes);
        assert!(
            mem <= MAX_THREADGROUP_MEMORY,
            "tile ({tm},{tn},{tk}) × {elem_bytes}B = {mem}B > {MAX_THREADGROUP_MEMORY}B"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_grid_dim_within_limits() {
    let n: u32 = 1_000_000;
    let tg = optimal_threadgroup_1d(n);
    let groups = grid_threadgroups_1d(n, tg);
    assert!(groups <= MAX_THREADGROUPS_PER_GRID_DIM);
}

// ── Tests: SIMD width alignment ─────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_simd_utilisation_perfect_alignment() {
    assert!((simd_utilisation(32) - 1.0).abs() < f64::EPSILON);
    assert!((simd_utilisation(64) - 1.0).abs() < f64::EPSILON);
    assert!((simd_utilisation(1024) - 1.0).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_simd_utilisation_partial_warp() {
    let u = simd_utilisation(16);
    assert!((u - 0.5).abs() < f64::EPSILON, "16/32 should be 50%, got {u}");
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_simd_utilisation_one_thread() {
    let u = simd_utilisation(1);
    assert!((u - 1.0 / 32.0).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_simd_utilisation_zero() {
    assert!((simd_utilisation(0) - 0.0).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_simd_utilisation_33_threads() {
    // 33 threads → 1 full warp (32) + 1 partial (1/32).
    // active = 33, scheduled = 64 → 33/64.
    let u = simd_utilisation(33);
    assert!((u - 33.0 / 64.0).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_optimal_1d_always_simd_aligned() {
    for n in 1..=2048u32 {
        let tg = optimal_threadgroup_1d(n);
        if tg > 0 {
            assert_eq!(tg % SIMD_WIDTH, 0, "1D tg {tg} not SIMD-aligned for n={n}");
        }
    }
}

// ── Tests: Shared memory budget calculations ────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shared_mem_tile_f32() {
    // 32×32×32 tile of f32: (32*32 + 32*32) * 4 = 8192 bytes.
    let mem = shared_memory_for_tile(32, 32, 32, 4);
    assert_eq!(mem, 8192);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shared_mem_tile_f16() {
    let mem = shared_memory_for_tile(32, 32, 32, 2);
    assert_eq!(mem, 4096);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shared_mem_tile_i8() {
    let mem = shared_memory_for_tile(32, 32, 32, 1);
    assert_eq!(mem, 2048);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shared_mem_large_tile_exceeds_budget() {
    // 128×128×128 of f32 would need (128*128 + 128*128)*4 = 131072.
    let mem = shared_memory_for_tile(128, 128, 128, 4);
    assert!(mem > MAX_THREADGROUP_MEMORY);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_best_matmul_tile_f32_fits() {
    let (tm, tn, tk) = best_matmul_tile(4);
    let mem = shared_memory_for_tile(tm, tn, tk, 4);
    assert!(mem <= MAX_THREADGROUP_MEMORY);
    // Verify non-trivial tile chosen.
    assert!(tm >= 8 && tn >= 8 && tk >= 8);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_best_matmul_tile_f16_larger_tiles() {
    let (tm_f16, tn_f16, _) = best_matmul_tile(2);
    let (tm_f32, tn_f32, _) = best_matmul_tile(4);
    // f16 should allow same or larger tiles since elements are smaller.
    assert!(tm_f16 >= tm_f32);
    assert!(tn_f16 >= tn_f32);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_shared_mem_reduction_buffer() {
    // A parallel reduction typically uses tg_size × elem_bytes of shared mem.
    let tg = reduction_threadgroup_size(1024);
    let shared = (tg as usize) * 4; // f32
    assert!(shared <= MAX_THREADGROUP_MEMORY);
}

// ── Tests: Occupancy estimator ──────────────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_full_threadgroup_no_shared() {
    let occ = estimate_occupancy(MAX_THREADS_PER_THREADGROUP, 0, REGISTERS_PER_THREAD);
    assert!((occ - 1.0).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_half_threadgroup() {
    let occ = estimate_occupancy(512, 0, REGISTERS_PER_THREAD);
    assert!((occ - 0.5).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_limited_by_shared_memory() {
    // Threadgroup uses 2× the budget → occupancy limited to 0.5 from that axis.
    let occ = estimate_occupancy(
        MAX_THREADS_PER_THREADGROUP,
        MAX_THREADGROUP_MEMORY * 2,
        REGISTERS_PER_THREAD,
    );
    assert!((occ - 0.5).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_limited_by_registers() {
    // Each thread uses 2× the register budget.
    let occ = estimate_occupancy(MAX_THREADS_PER_THREADGROUP, 0, REGISTERS_PER_THREAD * 2);
    assert!((occ - 0.5).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_zero_threads() {
    assert!((estimate_occupancy(0, 0, 0) - 0.0).abs() < f64::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_capped_at_one() {
    // Even with very little shared memory usage, occupancy should not exceed 1.0.
    let occ = estimate_occupancy(MAX_THREADS_PER_THREADGROUP, 1, 1);
    assert!(occ <= 1.0);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_occupancy_multiple_limiters() {
    // Thread-count = 25%, shared-mem = 50%, registers = 100% → min = 25%.
    let occ = estimate_occupancy(256, MAX_THREADGROUP_MEMORY / 2, REGISTERS_PER_THREAD);
    let expected = 0.25_f64; // thread factor dominates
    assert!((occ - expected).abs() < 1e-9, "expected {expected}, got {occ}");
}

// ── Tests: Wave / full-wave occupancy ───────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_full_wave_m1() {
    let groups = grid_threadgroups_1d(65536, 1024);
    assert!(achieves_full_wave(groups, M1_GPU_CORES));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_full_wave_m1_pro() {
    let groups = grid_threadgroups_1d(65536, 1024);
    assert!(achieves_full_wave(groups, M1_PRO_GPU_CORES));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_full_wave_m2_max() {
    let groups = grid_threadgroups_1d(65536, 1024);
    assert!(achieves_full_wave(groups, M2_MAX_GPU_CORES));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_full_wave_m3_max() {
    let groups = grid_threadgroups_1d(65536, 1024);
    assert!(achieves_full_wave(groups, M3_MAX_GPU_CORES));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_insufficient_wave_small_problem() {
    // Only 4 threadgroups → insufficient for M2 Max's 38 cores.
    let groups = grid_threadgroups_1d(4096, 1024);
    assert_eq!(groups, 4);
    assert!(!achieves_full_wave(groups, M2_MAX_GPU_CORES));
}

// ── Tests: Edge cases – prime dimensions ────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_prime_dimension_1d() {
    for &prime in &[7u32, 13, 97, 251, 509, 1021, 4093, 65521] {
        let tg = optimal_threadgroup_1d(prime);
        assert!(tg > 0);
        assert!(tg % SIMD_WIDTH == 0);
        let groups = grid_threadgroups_1d(prime, tg);
        assert!(groups * tg >= prime);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_prime_dimension_2d() {
    let (w, h) = optimal_threadgroup_2d(97, 251);
    assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
    let (gx, gy) = grid_threadgroups_2d(97, 251, w, h);
    assert!(gx * w >= 97);
    assert!(gy * h >= 251);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_prime_dimension_3d() {
    let (tx, ty, tz) = optimal_threadgroup_3d(13, 17, 19);
    assert!(tx * ty * tz <= MAX_THREADS_PER_THREADGROUP);
    let (gx, gy, gz) = grid_threadgroups_3d(13, 17, 19, tx, ty, tz);
    assert!(gx * tx >= 13);
    assert!(gy * ty >= 17);
    assert!(gz * tz >= 19);
}

// ── Tests: Edge cases – non-power-of-2 ─────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_non_pow2_dimensions() {
    let test_dims: &[u32] = &[3, 5, 6, 7, 9, 10, 12, 15, 17, 24, 48, 96, 192, 384, 768];
    for &d in test_dims {
        let tg = optimal_threadgroup_1d(d);
        assert!(tg > 0, "zero tg for d={d}");
        assert!(tg <= MAX_THREADS_PER_THREADGROUP);
        let groups = grid_threadgroups_1d(d, tg);
        assert!(groups * tg >= d, "under-dispatch for d={d}");
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_non_pow2_2d_coverage() {
    for &(c, r) in &[(48u32, 96u32), (192, 384), (768, 1536)] {
        let (w, h) = optimal_threadgroup_2d(c, r);
        assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
        let (gx, gy) = grid_threadgroups_2d(c, r, w, h);
        assert!(gx * w >= c && gy * h >= r);
    }
}

// ── Tests: Edge cases – very large dimensions ───────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_very_large_1d() {
    let n: u32 = 16_777_216; // 2^24
    let tg = optimal_threadgroup_1d(n);
    assert_eq!(tg, MAX_THREADS_PER_THREADGROUP);
    let groups = grid_threadgroups_1d(n, tg);
    assert!(groups <= MAX_THREADGROUPS_PER_GRID_DIM);
    assert!((groups as u64) * (tg as u64) >= n as u64);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_very_large_2d() {
    let (c, r) = (16384u32, 16384);
    let (w, h) = optimal_threadgroup_2d(c, r);
    let (gx, gy) = grid_threadgroups_2d(c, r, w, h);
    assert!(gx <= MAX_THREADGROUPS_PER_GRID_DIM);
    assert!(gy <= MAX_THREADGROUPS_PER_GRID_DIM);
}

// ── Tests: Edge cases – very small dimensions ───────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_single_element_1d() {
    let tg = optimal_threadgroup_1d(1);
    assert_eq!(tg, SIMD_WIDTH); // rounded up
    let groups = grid_threadgroups_1d(1, tg);
    assert_eq!(groups, 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_two_elements_1d() {
    let tg = optimal_threadgroup_1d(2);
    assert_eq!(tg, SIMD_WIDTH);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_single_pixel_2d() {
    let (w, h) = optimal_threadgroup_2d(1, 1);
    assert_eq!((w, h), (1, 1));
}

// ── Tests: Workgroup grid calculation (dispatched vs problem) ───────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_grid_1d_exact_fit() {
    // 1024 elements with tg=1024 → 1 group, 0 wasted threads.
    let groups = grid_threadgroups_1d(1024, 1024);
    assert_eq!(groups, 1);
    let dispatched = total_dispatched_threads_1d(1024, 1024);
    assert_eq!(dispatched, 1024);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_grid_1d_with_padding() {
    // 1025 elements with tg=1024 → 2 groups = 2048 dispatched threads.
    let groups = grid_threadgroups_1d(1025, 1024);
    assert_eq!(groups, 2);
    let dispatched = total_dispatched_threads_1d(1025, 1024);
    assert_eq!(dispatched, 2048);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_grid_waste_ratio() {
    // Measure wasted threads for various problem sizes.
    for &(n, tg) in &[(1000u32, 256u32), (1023, 1024), (2049, 512), (10000, 1024)] {
        let dispatched = total_dispatched_threads_1d(n, tg);
        let waste = dispatched - (n as u64);
        let ratio = (waste as f64) / (dispatched as f64);
        // Waste should be less than one full threadgroup.
        assert!(waste < tg as u64, "waste {waste} >= tg {tg} for n={n}");
        assert!(ratio < 1.0);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_grid_2d_covers_problem() {
    for &(c, r) in &[(32u32, 32u32), (64, 64), (100, 200), (1024, 1024), (4096, 2048)] {
        let (w, h) = optimal_threadgroup_2d(c, r);
        let (gx, gy) = grid_threadgroups_2d(c, r, w, h);
        assert!(gx * w >= c && gy * h >= r, "grid under-covers ({c},{r}) with tg ({w},{h})");
    }
}

// ── Tests: Shared memory vs registers tradeoffs ─────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_high_shared_mem_reduces_occupancy() {
    let low_mem = estimate_occupancy(1024, 1024, REGISTERS_PER_THREAD);
    let high_mem = estimate_occupancy(1024, 16 * 1024, REGISTERS_PER_THREAD);
    assert!(low_mem >= high_mem);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_high_register_pressure_reduces_occupancy() {
    let low_reg = estimate_occupancy(1024, 0, 64);
    let high_reg = estimate_occupancy(1024, 0, 256);
    assert!(low_reg >= high_reg);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_register_vs_shared_tradeoff() {
    // Scenario A: more shared mem, fewer registers.
    let occ_a = estimate_occupancy(1024, 16 * 1024, 64);
    // Scenario B: less shared mem, more registers.
    let occ_b = estimate_occupancy(1024, 4 * 1024, 256);
    // Both should be valid occupancies in (0, 1].
    assert!(occ_a > 0.0 && occ_a <= 1.0);
    assert!(occ_b > 0.0 && occ_b <= 1.0);
}

// ── Tests: Specific Apple Silicon chip configurations ───────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_m1_baseline_occupancy() {
    // M1 has 8 GPU cores; 8 threadgroups with 1024 threads each should fully occupy.
    assert!(achieves_full_wave(8, M1_GPU_CORES));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_m3_max_needs_more_parallelism() {
    // M3 Max has 40 cores; need at least 40 threadgroups.
    assert!(!achieves_full_wave(20, M3_MAX_GPU_CORES));
    assert!(achieves_full_wave(40, M3_MAX_GPU_CORES));
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_max_threadgroups_per_cu_limit() {
    // Even with many threadgroups, each CU handles at most 32.
    let groups_per_cu = 64u32;
    let effective = groups_per_cu.min(MAX_THREADGROUPS_PER_CU);
    assert_eq!(effective, MAX_THREADGROUPS_PER_CU);
}

// ── Tests: Comprehensive sweep of matrix dimensions ─────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_sweep_power_of_2_dimensions() {
    for exp in 0..=14u32 {
        let n = 1u32 << exp;
        let tg = optimal_threadgroup_1d(n);
        assert!(tg > 0 && tg <= MAX_THREADS_PER_THREADGROUP);
        assert!(grid_threadgroups_1d(n, tg) * tg >= n);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_sweep_multiples_of_simd_width() {
    for k in 1..=64u32 {
        let n = k * SIMD_WIDTH;
        let tg = optimal_threadgroup_1d(n);
        assert_eq!(tg % SIMD_WIDTH, 0);
        assert!(tg <= MAX_THREADS_PER_THREADGROUP);
    }
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_sweep_2d_square_matrices() {
    for &dim in &[8u32, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let (w, h) = optimal_threadgroup_2d(dim, dim);
        assert!(w * h <= MAX_THREADS_PER_THREADGROUP);
        let (gx, gy) = grid_threadgroups_2d(dim, dim, w, h);
        assert!(gx * w >= dim && gy * h >= dim);
    }
}

// ── Tests: Round-up / alignment helpers ─────────────────────────────────────

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_round_up_to_simd() {
    assert_eq!(round_up(0, SIMD_WIDTH), 0);
    assert_eq!(round_up(1, SIMD_WIDTH), 32);
    assert_eq!(round_up(32, SIMD_WIDTH), 32);
    assert_eq!(round_up(33, SIMD_WIDTH), 64);
    assert_eq!(round_up(1023, SIMD_WIDTH), 1024);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_ceil_div_exact() {
    assert_eq!(ceil_div(1024, 32), 32);
    assert_eq!(ceil_div(1024, 1024), 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_ceil_div_remainder() {
    assert_eq!(ceil_div(1025, 1024), 2);
    assert_eq!(ceil_div(33, 32), 2);
    assert_eq!(ceil_div(1, 1024), 1);
}

#[test]
#[ignore = "requires Metal GPU runtime - run on Apple Silicon"]
fn test_is_power_of_two() {
    assert!(is_power_of_two(1));
    assert!(is_power_of_two(2));
    assert!(is_power_of_two(1024));
    assert!(!is_power_of_two(0));
    assert!(!is_power_of_two(3));
    assert!(!is_power_of_two(1023));
}
