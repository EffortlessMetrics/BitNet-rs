#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types)]
//! Metal workgroup optimization validation tests for Apple Silicon.
//!
//! Validates workgroup size selection, dispatch grid calculation,
//! occupancy optimization, SIMD group operations, performance
//! heuristics, and dynamic dispatch tuning.  Pure Rust mocks — no
//! Metal SDK required.  All tests run on both aarch64 and x86_64.
#![cfg(feature = "cpu")]

// ── Apple Silicon Metal constants ──────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon (M1–M4).
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD (warp) width on Apple Silicon GPUs.
const SIMD_WIDTH: u32 = 32;

/// Maximum threadgroup memory (shared/tile memory) in bytes.
const MAX_THREADGROUP_MEMORY: usize = 32_768;

/// Maximum dispatch dimension per axis (Metal limit).
const MAX_DISPATCH_DIM: u32 = 65_535;

/// Metal buffer alignment in bytes.
const BUFFER_ALIGNMENT: usize = 256;

/// Typical register file capacity per thread (bytes, approximate).
const REGISTERS_PER_THREAD: usize = 128;

/// Maximum concurrent threadgroups per compute unit.
const MAX_THREADGROUPS_PER_CU: u32 = 32;

// ── Chip core counts ───────────────────────────────────────────────

const M1_GPU_CORES: u32 = 8;
const M1_PRO_GPU_CORES: u32 = 16;
const M2_GPU_CORES: u32 = 10;
const M2_MAX_GPU_CORES: u32 = 38;
const M3_GPU_CORES: u32 = 10;
const M3_MAX_GPU_CORES: u32 = 40;

// ── Mock types ─────────────────────────────────────────────────────

/// Workgroup (threadgroup) size along three axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl WorkgroupSize {
    fn new(x: u32, y: u32, z: u32) -> Option<Self> {
        let total = (x as u64) * (y as u64) * (z as u64);
        if x == 0 || y == 0 || z == 0 || total > MAX_THREADS_PER_THREADGROUP as u64 {
            return None;
        }
        Some(Self { x, y, z })
    }

    fn linear(n: u32) -> Option<Self> {
        Self::new(n, 1, 1)
    }

    fn tile(size: u32) -> Option<Self> {
        Self::new(size, size, 1)
    }

    fn total_threads(self) -> u32 {
        self.x * self.y * self.z
    }

    fn is_simd_aligned(self) -> bool {
        self.total_threads().is_multiple_of(SIMD_WIDTH)
    }

    fn simd_groups(self) -> u32 {
        self.total_threads().div_ceil(SIMD_WIDTH)
    }
}

/// Dispatch grid dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchGrid {
    x: u32,
    y: u32,
    z: u32,
}

impl DispatchGrid {
    fn for_problem(problem: (u32, u32, u32), wg: &WorkgroupSize) -> Option<Self> {
        let dx = problem.0.div_ceil(wg.x);
        let dy = problem.1.div_ceil(wg.y);
        let dz = problem.2.div_ceil(wg.z);
        if dx > MAX_DISPATCH_DIM || dy > MAX_DISPATCH_DIM || dz > MAX_DISPATCH_DIM {
            return None;
        }
        Some(Self { x: dx, y: dy, z: dz })
    }

    fn total_groups(self) -> u64 {
        (self.x as u64) * (self.y as u64) * (self.z as u64)
    }

    fn total_threads(self, wg: &WorkgroupSize) -> u64 {
        self.total_groups() * wg.total_threads() as u64
    }
}

/// Apple Silicon chip variant for tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppleChip {
    M1,
    M1Pro,
    M2,
    M2Max,
    M3,
    M3Max,
}

impl AppleChip {
    fn gpu_cores(self) -> u32 {
        match self {
            Self::M1 => M1_GPU_CORES,
            Self::M1Pro => M1_PRO_GPU_CORES,
            Self::M2 => M2_GPU_CORES,
            Self::M2Max => M2_MAX_GPU_CORES,
            Self::M3 => M3_GPU_CORES,
            Self::M3Max => M3_MAX_GPU_CORES,
        }
    }

    fn max_concurrent_threadgroups(self) -> u32 {
        self.gpu_cores() * MAX_THREADGROUPS_PER_CU
    }
}

/// Operation kind used for workgroup selection heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpKind {
    MatMul,
    Attention,
    ElementWise,
    Reduction,
    Convolution,
    Softmax,
}

/// Occupancy estimator.
#[derive(Debug, Clone, Copy)]
struct OccupancyEstimate {
    threads_per_group: u32,
    shared_memory_bytes: usize,
    registers_per_thread: usize,
    active_groups_per_cu: u32,
    occupancy_pct: f64,
}

/// Tile configuration for matmul heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TileConfig {
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
}

// ── Helper functions ───────────────────────────────────────────────

fn ceil_div(a: u32, b: u32) -> u32 {
    assert_ne!(b, 0);
    a.div_ceil(b)
}

fn is_power_of_two(n: u32) -> bool {
    n != 0 && n.is_power_of_two()
}

/// Choose an optimal workgroup for `op` with the given problem dims.
fn select_workgroup(op: OpKind, m: u32, n: u32) -> WorkgroupSize {
    match op {
        OpKind::MatMul => {
            // Prefer largest square tile that fits.
            [32, 16, 8]
                .iter()
                .copied()
                .find(|&t| t * t <= MAX_THREADS_PER_THREADGROUP)
                .and_then(|t| WorkgroupSize::tile(t))
                .unwrap()
        }
        OpKind::Attention => {
            // Head-dim-aligned 1-D group, capped.
            let width = n.min(MAX_THREADS_PER_THREADGROUP);
            let width = (width / SIMD_WIDTH) * SIMD_WIDTH;
            let width = width.max(SIMD_WIDTH);
            WorkgroupSize::linear(width).unwrap()
        }
        OpKind::ElementWise => {
            let total = (m as u64) * (n as u64);
            let width = if total >= 1024 {
                256
            } else if total >= 256 {
                128
            } else {
                64
            };
            WorkgroupSize::linear(width).unwrap()
        }
        OpKind::Reduction => {
            // Power-of-two width for efficient tree reduction.
            let width = n.next_power_of_two().min(MAX_THREADS_PER_THREADGROUP);
            WorkgroupSize::linear(width).unwrap()
        }
        OpKind::Convolution => {
            // 2-D tile, balance spatial coverage.
            let ty = m.min(8);
            let tx = (MAX_THREADS_PER_THREADGROUP / ty).min(n);
            WorkgroupSize::new(tx, ty, 1).unwrap()
        }
        OpKind::Softmax => {
            let width = n.next_power_of_two().min(MAX_THREADS_PER_THREADGROUP);
            let width = (width / SIMD_WIDTH) * SIMD_WIDTH;
            let width = width.max(SIMD_WIDTH);
            WorkgroupSize::linear(width).unwrap()
        }
    }
}

/// Estimate occupancy for a given workgroup config.
fn estimate_occupancy(wg: &WorkgroupSize, shared_mem: usize, regs: usize) -> OccupancyEstimate {
    let threads = wg.total_threads();
    // Shared-memory–limited groups per CU.
    let groups_by_mem = if shared_mem == 0 {
        MAX_THREADGROUPS_PER_CU
    } else {
        (MAX_THREADGROUP_MEMORY / shared_mem) as u32
    };
    // Thread-count–limited groups per CU.
    let groups_by_threads = MAX_THREADS_PER_THREADGROUP / threads;
    let active = groups_by_mem.min(groups_by_threads).min(MAX_THREADGROUPS_PER_CU);
    let active_threads = active * threads;
    let max_threads = MAX_THREADGROUPS_PER_CU * MAX_THREADS_PER_THREADGROUP;
    let pct = (active_threads as f64 / max_threads as f64) * 100.0;
    OccupancyEstimate {
        threads_per_group: threads,
        shared_memory_bytes: shared_mem,
        registers_per_thread: regs,
        active_groups_per_cu: active,
        occupancy_pct: pct,
    }
}

/// Choose a matmul tile configuration for given dims.
fn select_matmul_tile(m: u32, n: u32, k: u32) -> TileConfig {
    if m >= 128 && n >= 128 {
        TileConfig { tile_m: 32, tile_n: 32, tile_k: 8 }
    } else if m >= 32 && n >= 32 {
        TileConfig { tile_m: 16, tile_n: 16, tile_k: 8 }
    } else {
        TileConfig { tile_m: 8, tile_n: 8, tile_k: k.min(8) }
    }
}

/// Check whether a 2-D access pattern coalesces well.
fn is_coalesced(row_stride: usize, elem_bytes: usize) -> bool {
    let access_width = SIMD_WIDTH as usize * elem_bytes;
    row_stride.is_multiple_of(access_width)
}

/// Detect potential shared-memory bank conflicts.
fn has_bank_conflict(stride: usize, num_banks: usize) -> bool {
    stride.is_multiple_of(num_banks)
}

/// Dynamic dispatch: choose workgroup at runtime based on size.
fn dynamic_workgroup(op: OpKind, total_elements: u64) -> WorkgroupSize {
    match op {
        OpKind::ElementWise | OpKind::Softmax => {
            if total_elements >= 1_000_000 {
                WorkgroupSize::linear(256).unwrap()
            } else if total_elements >= 10_000 {
                WorkgroupSize::linear(128).unwrap()
            } else {
                WorkgroupSize::linear(64).unwrap()
            }
        }
        OpKind::MatMul => {
            if total_elements >= 65_536 {
                WorkgroupSize::tile(32).unwrap()
            } else if total_elements >= 1024 {
                WorkgroupSize::tile(16).unwrap()
            } else {
                WorkgroupSize::tile(8).unwrap()
            }
        }
        _ => WorkgroupSize::linear(
            (total_elements as u32)
                .next_power_of_two()
                .clamp(SIMD_WIDTH, MAX_THREADS_PER_THREADGROUP),
        )
        .unwrap(),
    }
}

/// Cache-friendly dispatch ordering score (lower = better).
fn cache_friendliness_score(wg: &WorkgroupSize, problem_n: u32) -> u32 {
    // Prefer workgroup widths that divide the problem width
    // for coalesced memory access.
    if problem_n.is_multiple_of(wg.x) { 0 } else { problem_n % wg.x }
}

/// SIMD reduction result for `n` elements.
fn simd_reduction_steps(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    // Tree reduction: log2(ceil(n / SIMD_WIDTH)) inter-SIMD
    // steps plus intra-SIMD shuffle.
    let groups = ceil_div(n, SIMD_WIDTH);
    let inter = (groups as f64).log2().ceil() as u32;
    let intra = (SIMD_WIDTH as f64).log2() as u32;
    inter + intra
}

/// Align `size` up to `BUFFER_ALIGNMENT`.
fn align_to_buffer(size: usize) -> usize {
    let mask = BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

// ===================================================================
// 1. Workgroup Size Selection Tests
// ===================================================================

mod workgroup_size_selection {
    use super::*;

    #[test]
    fn matmul_8x8_tile() {
        let wg = WorkgroupSize::tile(8).unwrap();
        assert_eq!(wg.total_threads(), 64);
        assert!(wg.is_simd_aligned());
    }

    #[test]
    fn matmul_16x16_tile() {
        let wg = WorkgroupSize::tile(16).unwrap();
        assert_eq!(wg.total_threads(), 256);
        assert!(wg.is_simd_aligned());
    }

    #[test]
    fn matmul_32x32_tile() {
        let wg = WorkgroupSize::tile(32).unwrap();
        assert_eq!(wg.total_threads(), 1024);
        assert_eq!(wg.total_threads(), MAX_THREADS_PER_THREADGROUP);
    }

    #[test]
    fn tile_33_exceeds_limit() {
        assert!(WorkgroupSize::tile(33).is_none());
    }

    #[test]
    fn attention_head_dim_64() {
        let wg = select_workgroup(OpKind::Attention, 1, 64);
        assert!(wg.total_threads().is_multiple_of(SIMD_WIDTH));
        assert!(wg.total_threads() <= MAX_THREADS_PER_THREADGROUP);
    }

    #[test]
    fn attention_head_dim_128() {
        let wg = select_workgroup(OpKind::Attention, 1, 128);
        assert_eq!(wg.total_threads(), 128);
        assert!(wg.is_simd_aligned());
    }

    #[test]
    fn elementwise_small_problem() {
        let wg = select_workgroup(OpKind::ElementWise, 4, 4);
        assert_eq!(wg.total_threads(), 64);
    }

    #[test]
    fn elementwise_large_problem() {
        let wg = select_workgroup(OpKind::ElementWise, 1024, 1024);
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn reduction_power_of_two() {
        let wg = select_workgroup(OpKind::Reduction, 1, 512);
        assert!(is_power_of_two(wg.total_threads()));
    }

    #[test]
    fn reduction_non_power_rounds_up() {
        let wg = select_workgroup(OpKind::Reduction, 1, 300);
        assert!(is_power_of_two(wg.total_threads()));
        assert!(wg.total_threads() >= 300);
    }

    #[test]
    fn linear_workgroup_256() {
        let wg = WorkgroupSize::linear(256).unwrap();
        assert_eq!(wg.x, 256);
        assert_eq!(wg.y, 1);
        assert_eq!(wg.z, 1);
    }

    #[test]
    fn zero_dimension_rejected() {
        assert!(WorkgroupSize::new(0, 16, 1).is_none());
        assert!(WorkgroupSize::new(16, 0, 1).is_none());
        assert!(WorkgroupSize::new(16, 16, 0).is_none());
    }

    #[test]
    fn m1_matmul_selection() {
        let wg = select_workgroup(OpKind::MatMul, 1024, 1024);
        assert!(wg.total_threads() >= 64);
        assert!(wg.total_threads() <= MAX_THREADS_PER_THREADGROUP);
    }

    #[test]
    fn m2_elementwise_selection() {
        let wg = select_workgroup(OpKind::ElementWise, 2048, 2048);
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn m3_attention_large_head() {
        let wg = select_workgroup(OpKind::Attention, 1, 256);
        assert_eq!(wg.total_threads(), 256);
        assert!(wg.is_simd_aligned());
    }

    #[test]
    fn softmax_workgroup_simd_aligned() {
        let wg = select_workgroup(OpKind::Softmax, 1, 100);
        assert!(wg.is_simd_aligned());
    }

    #[test]
    fn convolution_2d_tile() {
        let wg = select_workgroup(OpKind::Convolution, 8, 128);
        assert!(wg.total_threads() <= MAX_THREADS_PER_THREADGROUP);
        assert!(wg.y <= 8);
    }
}

// ===================================================================
// 2. Dispatch Grid Tests
// ===================================================================

mod dispatch_grid {
    use super::*;

    #[test]
    fn exact_division_no_remainder() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let grid = DispatchGrid::for_problem((256, 256, 1), &wg).unwrap();
        assert_eq!(grid.x, 16);
        assert_eq!(grid.y, 16);
        assert_eq!(grid.z, 1);
    }

    #[test]
    fn remainder_rounds_up() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let grid = DispatchGrid::for_problem((257, 257, 1), &wg).unwrap();
        assert_eq!(grid.x, 17);
        assert_eq!(grid.y, 17);
    }

    #[test]
    fn single_element_problem() {
        let wg = WorkgroupSize::linear(64).unwrap();
        let grid = DispatchGrid::for_problem((1, 1, 1), &wg).unwrap();
        assert_eq!(grid.x, 1);
    }

    #[test]
    fn dispatch_1d_large() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let grid = DispatchGrid::for_problem((65536, 1, 1), &wg).unwrap();
        assert_eq!(grid.x, 256);
        assert_eq!(grid.total_groups(), 256);
    }

    #[test]
    fn dispatch_2d_grid() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let grid = DispatchGrid::for_problem((512, 512, 1), &wg).unwrap();
        assert_eq!(grid.x, 32);
        assert_eq!(grid.y, 32);
        assert_eq!(grid.total_groups(), 1024);
    }

    #[test]
    fn dispatch_3d_grid() {
        let wg = WorkgroupSize::new(8, 8, 4).unwrap();
        let grid = DispatchGrid::for_problem((64, 64, 16), &wg).unwrap();
        assert_eq!(grid.x, 8);
        assert_eq!(grid.y, 8);
        assert_eq!(grid.z, 4);
    }

    #[test]
    fn total_threads_covers_problem() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let problem = (100, 200, 1);
        let grid = DispatchGrid::for_problem(problem, &wg).unwrap();
        let threads = grid.total_threads(&wg);
        assert!(threads >= (problem.0 as u64) * (problem.1 as u64));
    }

    #[test]
    fn exceeds_max_dispatch_dim() {
        let wg = WorkgroupSize::linear(1).unwrap();
        let grid = DispatchGrid::for_problem((70_000, 1, 1), &wg);
        assert!(grid.is_none());
    }

    #[test]
    fn dispatch_grid_z_1_for_2d_problem() {
        let wg = WorkgroupSize::tile(8).unwrap();
        let grid = DispatchGrid::for_problem((64, 64, 1), &wg).unwrap();
        assert_eq!(grid.z, 1);
    }

    #[test]
    fn non_divisible_problem_x() {
        let wg = WorkgroupSize::linear(32).unwrap();
        let grid = DispatchGrid::for_problem((33, 1, 1), &wg).unwrap();
        assert_eq!(grid.x, 2);
    }

    #[test]
    fn non_divisible_problem_y() {
        let wg = WorkgroupSize::new(1, 16, 1).unwrap();
        let grid = DispatchGrid::for_problem((1, 17, 1), &wg).unwrap();
        assert_eq!(grid.y, 2);
    }

    #[test]
    fn non_divisible_problem_z() {
        let wg = WorkgroupSize::new(1, 1, 4).unwrap();
        let grid = DispatchGrid::for_problem((1, 1, 5), &wg).unwrap();
        assert_eq!(grid.z, 2);
    }

    #[test]
    fn grid_matches_ceil_div() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let problem = (1000, 500, 1);
        let grid = DispatchGrid::for_problem(problem, &wg).unwrap();
        assert_eq!(grid.x, ceil_div(1000, 16));
        assert_eq!(grid.y, ceil_div(500, 16));
    }

    #[test]
    fn large_2d_problem() {
        let wg = WorkgroupSize::tile(32).unwrap();
        let grid = DispatchGrid::for_problem((4096, 4096, 1), &wg).unwrap();
        assert_eq!(grid.x, 128);
        assert_eq!(grid.y, 128);
    }

    #[test]
    fn batch_dimension_dispatch() {
        let wg = WorkgroupSize::new(16, 16, 1).unwrap();
        let batch = 8;
        let grid = DispatchGrid::for_problem((256, 256, batch), &wg).unwrap();
        assert_eq!(grid.z, 8);
    }
}

// ===================================================================
// 3. Occupancy Optimization Tests
// ===================================================================

mod occupancy_optimization {
    use super::*;

    #[test]
    fn full_occupancy_small_workgroup() {
        let wg = WorkgroupSize::linear(32).unwrap();
        let est = estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD);
        assert!(est.occupancy_pct > 0.0);
        assert!(est.active_groups_per_cu > 0);
    }

    #[test]
    fn shared_memory_limits_occupancy() {
        let wg = WorkgroupSize::linear(256).unwrap();
        // Use half of shared memory → at most 2 groups.
        let est = estimate_occupancy(&wg, MAX_THREADGROUP_MEMORY / 2, REGISTERS_PER_THREAD);
        assert!(est.active_groups_per_cu <= 2);
    }

    #[test]
    fn zero_shared_memory_max_groups() {
        let wg = WorkgroupSize::linear(32).unwrap();
        let est = estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD);
        assert_eq!(est.active_groups_per_cu, MAX_THREADGROUPS_PER_CU);
    }

    #[test]
    fn max_workgroup_limits_groups() {
        let wg = WorkgroupSize::tile(32).unwrap();
        assert_eq!(wg.total_threads(), 1024);
        let est = estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD);
        assert_eq!(est.active_groups_per_cu, 1);
    }

    #[test]
    fn occupancy_pct_range() {
        let wg = WorkgroupSize::linear(128).unwrap();
        let est = estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD);
        assert!(est.occupancy_pct > 0.0);
        assert!(est.occupancy_pct <= 100.0);
    }

    #[test]
    fn large_shared_memory_low_occupancy() {
        let wg = WorkgroupSize::linear(64).unwrap();
        let est = estimate_occupancy(&wg, MAX_THREADGROUP_MEMORY, REGISTERS_PER_THREAD);
        assert_eq!(est.active_groups_per_cu, 1);
    }

    #[test]
    fn register_pressure_recorded() {
        let wg = WorkgroupSize::linear(64).unwrap();
        let est = estimate_occupancy(&wg, 1024, REGISTERS_PER_THREAD);
        assert_eq!(est.registers_per_thread, REGISTERS_PER_THREAD);
    }

    #[test]
    fn apple_max_1024_threads() {
        assert!(WorkgroupSize::linear(1024).is_some());
        assert!(WorkgroupSize::linear(1025).is_none());
    }

    #[test]
    fn occupancy_decreases_with_shared_mem() {
        let wg = WorkgroupSize::linear(64).unwrap();
        let low = estimate_occupancy(&wg, 1024, REGISTERS_PER_THREAD);
        let high = estimate_occupancy(&wg, MAX_THREADGROUP_MEMORY / 2, REGISTERS_PER_THREAD);
        assert!(low.active_groups_per_cu >= high.active_groups_per_cu);
    }

    #[test]
    fn optimal_workgroup_for_occupancy() {
        // Compare 64 vs 256 vs 1024 thread groups.
        let sizes = [64_u32, 256, 1024];
        let estimates: Vec<_> = sizes
            .iter()
            .map(|&s| {
                let wg = WorkgroupSize::linear(s).unwrap();
                estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD)
            })
            .collect();
        // Smallest workgroup allows most concurrent groups.
        assert!(estimates[0].active_groups_per_cu >= estimates[2].active_groups_per_cu);
    }

    #[test]
    fn threadgroup_memory_32k_limit() {
        assert_eq!(MAX_THREADGROUP_MEMORY, 32_768);
    }

    #[test]
    fn occupancy_with_4k_shared() {
        let wg = WorkgroupSize::linear(128).unwrap();
        let est = estimate_occupancy(&wg, 4096, REGISTERS_PER_THREAD);
        assert_eq!(
            est.active_groups_per_cu,
            (MAX_THREADGROUP_MEMORY / 4096).min((MAX_THREADS_PER_THREADGROUP / 128) as usize)
                as u32
        );
    }

    #[test]
    fn two_dim_workgroup_occupancy() {
        let wg = WorkgroupSize::new(16, 8, 1).unwrap();
        let est = estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD);
        assert_eq!(est.threads_per_group, 128);
        assert!(est.active_groups_per_cu > 0);
    }

    #[test]
    fn three_dim_workgroup_occupancy() {
        let wg = WorkgroupSize::new(8, 8, 4).unwrap();
        let est = estimate_occupancy(&wg, 0, REGISTERS_PER_THREAD);
        assert_eq!(est.threads_per_group, 256);
    }

    #[test]
    fn shared_memory_bytes_preserved() {
        let wg = WorkgroupSize::linear(64).unwrap();
        let est = estimate_occupancy(&wg, 8192, REGISTERS_PER_THREAD);
        assert_eq!(est.shared_memory_bytes, 8192);
    }
}

// ===================================================================
// 4. SIMD Group Tests
// ===================================================================

mod simd_group {
    use super::*;

    #[test]
    fn apple_simd_width_is_32() {
        assert_eq!(SIMD_WIDTH, 32);
    }

    #[test]
    fn simd_groups_32_threads() {
        let wg = WorkgroupSize::linear(32).unwrap();
        assert_eq!(wg.simd_groups(), 1);
    }

    #[test]
    fn simd_groups_256_threads() {
        let wg = WorkgroupSize::linear(256).unwrap();
        assert_eq!(wg.simd_groups(), 8);
    }

    #[test]
    fn simd_groups_1024_threads() {
        let wg = WorkgroupSize::linear(1024).unwrap();
        assert_eq!(wg.simd_groups(), 32);
    }

    #[test]
    fn simd_aligned_multiples() {
        [32_u32, 64, 128, 256, 512, 1024].iter().for_each(|&n| {
            let wg = WorkgroupSize::linear(n).unwrap();
            assert!(wg.is_simd_aligned(), "{n} should be SIMD-aligned");
        });
    }

    #[test]
    fn simd_not_aligned_odd_sizes() {
        [33_u32, 48, 100].iter().for_each(|&n| {
            if let Some(wg) = WorkgroupSize::linear(n) {
                assert!(!wg.is_simd_aligned(), "{n} should NOT be SIMD-aligned");
            }
        });
    }

    #[test]
    fn simd_reduction_steps_32() {
        let steps = simd_reduction_steps(32);
        // Intra-SIMD only: log2(32) = 5.
        assert_eq!(steps, 5);
    }

    #[test]
    fn simd_reduction_steps_1024() {
        let steps = simd_reduction_steps(1024);
        // 1024 / 32 = 32 SIMD groups → 5 inter + 5 intra = 10.
        assert_eq!(steps, 10);
    }

    #[test]
    fn simd_reduction_steps_1() {
        assert_eq!(simd_reduction_steps(1), 0);
    }

    #[test]
    fn simd_reduction_monotonic() {
        let steps: Vec<u32> =
            [32, 64, 128, 256, 512, 1024].iter().map(|&n| simd_reduction_steps(n)).collect();
        steps.windows(2).for_each(|w| {
            assert!(w[0] <= w[1], "reduction steps should be monotonic");
        });
    }

    #[test]
    fn simd_broadcast_one_group() {
        // Broadcast within one SIMD group: constant time.
        let wg = WorkgroupSize::linear(32).unwrap();
        assert_eq!(wg.simd_groups(), 1);
    }

    #[test]
    fn simd_groups_2d_tile() {
        let wg = WorkgroupSize::tile(16).unwrap();
        // 16×16 = 256, 256/32 = 8 SIMD groups.
        assert_eq!(wg.simd_groups(), 8);
    }

    #[test]
    fn simd_groups_non_multiple() {
        let wg = WorkgroupSize::linear(48).unwrap();
        // 48 / 32 = 1.5 → ceil = 2.
        assert_eq!(wg.simd_groups(), 2);
    }

    #[test]
    fn wave_level_prefix_sum_steps() {
        // Prefix sum in one SIMD group: log2(32) = 5 steps.
        let intra_steps = (SIMD_WIDTH as f64).log2() as u32;
        assert_eq!(intra_steps, 5);
    }

    #[test]
    fn all_standard_tiles_simd_aligned() {
        [8_u32, 16, 32].iter().for_each(|&s| {
            let wg = WorkgroupSize::tile(s).unwrap();
            assert!(wg.is_simd_aligned(), "tile {s}x{s} not SIMD-aligned");
        });
    }
}

// ===================================================================
// 5. Performance Heuristic Tests
// ===================================================================

mod performance_heuristics {
    use super::*;

    #[test]
    fn large_matmul_tile_32() {
        let tile = select_matmul_tile(1024, 1024, 64);
        assert_eq!(tile.tile_m, 32);
        assert_eq!(tile.tile_n, 32);
    }

    #[test]
    fn medium_matmul_tile_16() {
        let tile = select_matmul_tile(64, 64, 32);
        assert_eq!(tile.tile_m, 16);
        assert_eq!(tile.tile_n, 16);
    }

    #[test]
    fn small_matmul_tile_8() {
        let tile = select_matmul_tile(8, 8, 4);
        assert_eq!(tile.tile_m, 8);
        assert_eq!(tile.tile_n, 8);
    }

    #[test]
    fn tile_k_capped() {
        let tile = select_matmul_tile(4, 4, 2);
        assert!(tile.tile_k <= 8);
    }

    #[test]
    fn coalesced_access_float4() {
        // 32 threads × 4 bytes = 128-byte access line.
        assert!(is_coalesced(128, 4));
    }

    #[test]
    fn non_coalesced_access() {
        // Odd stride breaks coalescing.
        assert!(!is_coalesced(100, 4));
    }

    #[test]
    fn coalesced_access_half() {
        // 32 threads × 2 bytes = 64.
        assert!(is_coalesced(64, 2));
    }

    #[test]
    fn bank_conflict_stride_32() {
        // 32 banks → stride 32 hits same bank every time.
        assert!(has_bank_conflict(32, 32));
    }

    #[test]
    fn no_bank_conflict_stride_33() {
        assert!(!has_bank_conflict(33, 32));
    }

    #[test]
    fn bank_conflict_stride_64() {
        assert!(has_bank_conflict(64, 32));
    }

    #[test]
    fn buffer_alignment_256() {
        assert_eq!(align_to_buffer(1), BUFFER_ALIGNMENT);
        assert_eq!(align_to_buffer(256), 256);
        assert_eq!(align_to_buffer(257), 512);
    }

    #[test]
    fn tile_threads_within_limit() {
        [8_u32, 16, 32].iter().for_each(|&s| {
            let tile = select_matmul_tile(s * 4, s * 4, 8);
            let wg = WorkgroupSize::tile(tile.tile_m).unwrap();
            assert!(wg.total_threads() <= MAX_THREADS_PER_THREADGROUP);
        });
    }

    #[test]
    fn shared_memory_for_tile() {
        // tile_m × tile_k × sizeof(f32) for A tile.
        let tile = select_matmul_tile(1024, 1024, 64);
        let shared = (tile.tile_m as usize) * (tile.tile_k as usize) * 4;
        assert!(shared <= MAX_THREADGROUP_MEMORY);
    }

    #[test]
    fn coalesced_power_of_two_rows() {
        // Coalesced when stride % (SIMD_WIDTH * elem_bytes) == 0.
        // For f32 (4 bytes): 32 × 4 = 128.
        [128_usize, 256, 512, 1024].iter().for_each(|&stride| {
            assert!(is_coalesced(stride, 4), "stride {stride} not coalesced");
        });
    }

    #[test]
    fn matmul_tile_k_always_positive() {
        let tile = select_matmul_tile(1, 1, 1);
        assert!(tile.tile_k >= 1);
    }
}

// ===================================================================
// 6. Dynamic Dispatch Tests
// ===================================================================

mod dynamic_dispatch {
    use super::*;

    #[test]
    fn elementwise_tiny() {
        let wg = dynamic_workgroup(OpKind::ElementWise, 100);
        assert_eq!(wg.total_threads(), 64);
    }

    #[test]
    fn elementwise_medium() {
        let wg = dynamic_workgroup(OpKind::ElementWise, 50_000);
        assert_eq!(wg.total_threads(), 128);
    }

    #[test]
    fn elementwise_large() {
        let wg = dynamic_workgroup(OpKind::ElementWise, 2_000_000);
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn matmul_tiny() {
        let wg = dynamic_workgroup(OpKind::MatMul, 100);
        assert_eq!(wg.total_threads(), 64); // 8×8
    }

    #[test]
    fn matmul_medium() {
        let wg = dynamic_workgroup(OpKind::MatMul, 4096);
        assert_eq!(wg.total_threads(), 256); // 16×16
    }

    #[test]
    fn matmul_large() {
        let wg = dynamic_workgroup(OpKind::MatMul, 100_000);
        assert_eq!(wg.total_threads(), 1024); // 32×32
    }

    #[test]
    fn softmax_large() {
        let wg = dynamic_workgroup(OpKind::Softmax, 1_000_000);
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn reduction_dynamic_power_of_two() {
        let wg = dynamic_workgroup(OpKind::Reduction, 500);
        assert!(is_power_of_two(wg.total_threads()));
    }

    #[test]
    fn cache_friendly_score_exact() {
        let wg = WorkgroupSize::linear(32).unwrap();
        assert_eq!(cache_friendliness_score(&wg, 256), 0);
    }

    #[test]
    fn cache_friendly_score_remainder() {
        let wg = WorkgroupSize::linear(32).unwrap();
        let score = cache_friendliness_score(&wg, 100);
        assert_eq!(score, 100 % 32);
    }

    #[test]
    fn cache_friendliness_prefers_dividing() {
        let wg16 = WorkgroupSize::linear(16).unwrap();
        let wg32 = WorkgroupSize::linear(32).unwrap();
        // Problem width 48: 48 % 16 = 0, 48 % 32 = 16.
        assert!(cache_friendliness_score(&wg16, 48) < cache_friendliness_score(&wg32, 48));
    }

    #[test]
    fn dynamic_dispatch_always_valid() {
        [1_u64, 100, 10_000, 1_000_000, 100_000_000].iter().for_each(|&n| {
            let wg = dynamic_workgroup(OpKind::ElementWise, n);
            assert!(wg.total_threads() <= MAX_THREADS_PER_THREADGROUP);
            assert!(wg.total_threads() > 0);
        });
    }

    #[test]
    fn dynamic_dispatch_simd_aligned() {
        [1_000_u64, 50_000, 2_000_000].iter().for_each(|&n| {
            let wg = dynamic_workgroup(OpKind::ElementWise, n);
            assert!(wg.is_simd_aligned(), "n={n} produced non-aligned wg");
        });
    }

    #[test]
    fn chip_concurrent_threadgroups() {
        let chips = [
            AppleChip::M1,
            AppleChip::M1Pro,
            AppleChip::M2,
            AppleChip::M2Max,
            AppleChip::M3,
            AppleChip::M3Max,
        ];
        chips.iter().for_each(|chip| {
            let max_tg = chip.max_concurrent_threadgroups();
            assert!(
                max_tg >= M1_GPU_CORES * MAX_THREADGROUPS_PER_CU,
                "{chip:?} has fewer than M1 baseline"
            );
        });
    }

    #[test]
    fn m3_max_more_cores_than_m1() {
        assert!(AppleChip::M3Max.gpu_cores() > AppleChip::M1.gpu_cores());
    }
}

// ===================================================================
// Bonus: Cross-category integration-style tests
// ===================================================================

mod integration {
    use super::*;

    #[test]
    fn end_to_end_matmul_dispatch() {
        let wg = select_workgroup(OpKind::MatMul, 512, 512);
        let grid = DispatchGrid::for_problem((512, 512, 1), &wg).unwrap();
        let total_threads = grid.total_threads(&wg);
        assert!(total_threads >= 512 * 512);
    }

    #[test]
    fn end_to_end_attention_dispatch() {
        let seq_len: u32 = 2048;
        let head_dim: u32 = 64;
        let wg = select_workgroup(OpKind::Attention, seq_len, head_dim);
        let grid = DispatchGrid::for_problem(
            (seq_len, 1, 1),
            &WorkgroupSize::linear(wg.total_threads()).unwrap(),
        )
        .unwrap();
        assert!(
            grid.total_threads(&WorkgroupSize::linear(wg.total_threads()).unwrap())
                >= seq_len as u64
        );
    }

    #[test]
    fn occupancy_aware_tile_selection() {
        // Pick the tile that maximises occupancy.
        let best = [8_u32, 16, 32]
            .iter()
            .map(|&t| {
                let wg = WorkgroupSize::tile(t).unwrap();
                let est =
                    estimate_occupancy(&wg, (t as usize) * (t as usize) * 4, REGISTERS_PER_THREAD);
                (t, est.active_groups_per_cu)
            })
            .max_by_key(|&(_, g)| g)
            .unwrap();
        assert!(best.1 > 0);
    }

    #[test]
    fn dynamic_dispatch_grid_valid() {
        let wg = dynamic_workgroup(OpKind::ElementWise, 1_000_000);
        let grid = DispatchGrid::for_problem(
            (1_000_000, 1, 1),
            &WorkgroupSize::linear(wg.total_threads()).unwrap(),
        )
        .unwrap();
        assert!(grid.x <= MAX_DISPATCH_DIM);
    }

    #[test]
    fn full_pipeline_matmul_4096() {
        let m = 4096_u32;
        let n = 4096_u32;
        let k = 128_u32;
        let tile = select_matmul_tile(m, n, k);
        let wg = WorkgroupSize::tile(tile.tile_m).unwrap();
        let grid = DispatchGrid::for_problem((m, n, 1), &wg).unwrap();
        let est = estimate_occupancy(
            &wg,
            (tile.tile_m as usize) * (tile.tile_k as usize) * 4,
            REGISTERS_PER_THREAD,
        );
        assert!(grid.total_threads(&wg) >= (m as u64) * (n as u64));
        assert!(est.occupancy_pct > 0.0);
        assert!(wg.is_simd_aligned());
    }
}
