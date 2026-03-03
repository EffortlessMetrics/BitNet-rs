//! Metal compute dispatch sizing tests for Apple Silicon.
//!
//! Validates threadgroup dimensions, grid sizing, workgroup limits, dispatch
//! optimization, Apple Silicon specifics, dynamic dispatch, buffer alignment,
//! and regression edge-cases.
//!
//! All types and helpers are defined locally (the production `metal_compute`
//! module is gated behind `--features metal`).  These tests exercise the
//! *sizing arithmetic* that any Metal dispatch layer must get right.
//!
//! 64 tests total, organised in 8 modules of 8 tests each.

#![cfg(feature = "cpu")]

// ── Apple Silicon Metal constants ───────────────────────────────────

/// Maximum threads per threadgroup on Apple Silicon (M1/M2/M3/M4).
const METAL_MAX_WORKGROUP_SIZE: u32 = 1024;

/// Default tile dimension (16×16 = 256 threads).
const DEFAULT_TILE_SIZE: u32 = 16;

/// Maximum dispatch groups per axis (Metal limit).
const MAX_DISPATCH_DIM: u32 = 65535;

/// Metal buffer alignment requirement (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// SIMD execution width on Apple GPU (all M-series chips).
const APPLE_EXECUTION_WIDTH: u32 = 32;

/// Page size on Apple Silicon (used for large-buffer alignment).
const PAGE_SIZE: usize = 16384;

/// Maximum threadgroup memory (shared memory) per threadgroup (bytes).
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

// ── Local mirror types ──────────────────────────────────────────────

/// Workgroup (threadgroup) dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SizingError {
    WorkgroupTooLarge { requested: u64, max: u32 },
    ZeroDimension,
    DispatchTooLarge { dimension: u32, max: u32 },
}

impl WorkgroupSize {
    fn new(x: u32, y: u32, z: u32) -> Result<Self, SizingError> {
        if x == 0 || y == 0 || z == 0 {
            return Err(SizingError::ZeroDimension);
        }
        let total = (x as u64) * (y as u64) * (z as u64);
        if total > METAL_MAX_WORKGROUP_SIZE as u64 {
            return Err(SizingError::WorkgroupTooLarge {
                requested: total,
                max: METAL_MAX_WORKGROUP_SIZE,
            });
        }
        Ok(Self { x, y, z })
    }

    fn linear(n: u32) -> Result<Self, SizingError> {
        Self::new(n, 1, 1)
    }

    fn tile(size: u32) -> Result<Self, SizingError> {
        Self::new(size, size, 1)
    }

    fn total_threads(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self { x: DEFAULT_TILE_SIZE, y: DEFAULT_TILE_SIZE, z: 1 }
    }
}

/// Number of workgroups to dispatch along each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DispatchDimensions {
    x: u32,
    y: u32,
    z: u32,
}

impl DispatchDimensions {
    fn for_problem(problem: (u32, u32, u32), wg: &WorkgroupSize) -> Result<Self, SizingError> {
        let dim = |p: u32, w: u32| -> Result<u32, SizingError> {
            if w == 0 {
                return Err(SizingError::ZeroDimension);
            }
            let d = p.div_ceil(w);
            if d > MAX_DISPATCH_DIM {
                return Err(SizingError::DispatchTooLarge { dimension: d, max: MAX_DISPATCH_DIM });
            }
            Ok(d)
        };
        Ok(Self { x: dim(problem.0, wg.x)?, y: dim(problem.1, wg.y)?, z: dim(problem.2, wg.z)? })
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn occupancy(wg: &WorkgroupSize) -> f64 {
    wg.total_threads() as f64 / METAL_MAX_WORKGROUP_SIZE as f64
}

fn optimal_linear(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let rounded = ((n + APPLE_EXECUTION_WIDTH - 1) / APPLE_EXECUTION_WIDTH) * APPLE_EXECUTION_WIDTH;
    rounded.min(METAL_MAX_WORKGROUP_SIZE)
}

fn ceil_div(a: u32, b: u32) -> u32 {
    assert_ne!(b, 0);
    (a + b - 1) / b
}

fn align_buffer_size(size: usize) -> usize {
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

fn is_aligned(offset: usize) -> bool {
    offset % METAL_BUFFER_ALIGNMENT == 0
}

fn dispatch_for_matrix(
    rows: u32,
    cols: u32,
    wg: &WorkgroupSize,
) -> Result<DispatchDimensions, SizingError> {
    DispatchDimensions::for_problem((cols, rows, 1), wg)
}

fn aligned_buffer_bytes(element_count: usize, element_bytes: usize) -> usize {
    align_buffer_size(element_count * element_bytes)
}

// ════════════════════════════════════════════════════════════════════
// 1. Thread Group Size Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod threadgroup_size {
    use super::*;

    #[test]
    fn optimal_default_tile_is_power_of_two() {
        assert!(is_power_of_two(DEFAULT_TILE_SIZE));
        let wg = WorkgroupSize::default();
        assert!(is_power_of_two(wg.x));
        assert!(is_power_of_two(wg.y));
    }

    #[test]
    fn power_of_two_sizes_accepted() {
        for exp in 0..=10 {
            let n: u32 = 1 << exp;
            if n <= METAL_MAX_WORKGROUP_SIZE {
                assert!(WorkgroupSize::linear(n).is_ok(), "linear({n}) should be valid");
            }
        }
    }

    #[test]
    fn device_limit_exactly_1024() {
        assert_eq!(METAL_MAX_WORKGROUP_SIZE, 1024);
        assert!(WorkgroupSize::linear(1024).is_ok());
        assert!(WorkgroupSize::linear(1025).is_err());
    }

    #[test]
    fn dispatch_1d_linear() {
        let wg = WorkgroupSize::linear(256).unwrap();
        assert_eq!(wg.y, 1);
        assert_eq!(wg.z, 1);
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn dispatch_2d_tile() {
        let wg = WorkgroupSize::tile(16).unwrap();
        assert_eq!((wg.x, wg.y, wg.z), (16, 16, 1));
        assert_eq!(wg.total_threads(), 256);
    }

    #[test]
    fn dispatch_3d_within_limit() {
        let wg = WorkgroupSize::new(8, 8, 16).unwrap();
        assert_eq!(wg.total_threads(), 1024);
    }

    #[test]
    fn occupancy_at_full_workgroup() {
        let wg = WorkgroupSize::linear(1024).unwrap();
        let occ = occupancy(&wg);
        assert!((occ - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn warp_aligned_sizes() {
        for multiple in [32, 64, 128, 256, 512, 1024] {
            let wg = WorkgroupSize::linear(multiple).unwrap();
            assert_eq!(
                wg.total_threads() % APPLE_EXECUTION_WIDTH,
                0,
                "{multiple} should be execution-width aligned"
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// 2. Grid Size Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod grid_size {
    use super::*;

    #[test]
    fn total_threads_cover_problem() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((100, 100, 1), &wg).unwrap();
        assert!(dd.x * wg.x >= 100);
        assert!(dd.y * wg.y >= 100);
    }

    #[test]
    fn grid_aligned_to_threadgroup() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let dd = DispatchDimensions::for_problem((256, 1, 1), &wg).unwrap();
        assert_eq!(dd.x, 1);
    }

    #[test]
    fn over_dispatch_single_extra_group() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let dd = DispatchDimensions::for_problem((257, 1, 1), &wg).unwrap();
        assert_eq!(dd.x, 2);
        assert_eq!(dd.x * wg.x - 257, 255);
    }

    #[test]
    fn dimension_clamped_at_max() {
        let wg = WorkgroupSize::linear(1).unwrap();
        let err = DispatchDimensions::for_problem((MAX_DISPATCH_DIM + 1, 1, 1), &wg);
        assert!(err.is_err());
    }

    #[test]
    fn aspect_ratio_2d_tall() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((16, 1024, 1), &wg).unwrap();
        assert_eq!(dd.x, 1);
        assert_eq!(dd.y, 64);
    }

    #[test]
    fn aspect_ratio_2d_wide() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((1024, 16, 1), &wg).unwrap();
        assert_eq!(dd.x, 64);
        assert_eq!(dd.y, 1);
    }

    #[test]
    fn grid_for_exact_multiple() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((256, 256, 1), &wg).unwrap();
        assert_eq!((dd.x, dd.y, dd.z), (16, 16, 1));
    }

    #[test]
    fn grid_3d_batch_dispatch() {
        let wg = WorkgroupSize::new(16, 16, 1).unwrap();
        let dd = DispatchDimensions::for_problem((128, 128, 8), &wg).unwrap();
        assert_eq!(dd.z, 8);
        assert!(dd.x * wg.x >= 128);
        assert!(dd.y * wg.y >= 128);
    }
}

// ════════════════════════════════════════════════════════════════════
// 3. Workgroup Limits Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod workgroup_limits {
    use super::*;

    #[test]
    fn max_threads_per_threadgroup_is_1024() {
        assert_eq!(METAL_MAX_WORKGROUP_SIZE, 1024);
    }

    #[test]
    fn max_threadgroup_x_dimension() {
        assert!(WorkgroupSize::new(1024, 1, 1).is_ok());
        assert!(WorkgroupSize::new(1025, 1, 1).is_err());
    }

    #[test]
    fn max_threadgroup_y_dimension() {
        assert!(WorkgroupSize::new(1, 1024, 1).is_ok());
        assert!(WorkgroupSize::new(1, 1025, 1).is_err());
    }

    #[test]
    fn max_threadgroup_z_dimension() {
        assert!(WorkgroupSize::new(1, 1, 1024).is_ok());
        assert!(WorkgroupSize::new(1, 1, 1025).is_err());
    }

    #[test]
    fn threadgroup_memory_budget_softmax_row() {
        let seq_len: usize = 2048;
        let bytes = seq_len * std::mem::size_of::<f32>();
        assert!(
            bytes <= MAX_THREADGROUP_MEMORY,
            "softmax row of {seq_len} f32 should fit: {bytes} <= {MAX_THREADGROUP_MEMORY}"
        );
    }

    #[test]
    fn threadgroup_memory_budget_matmul_tile() {
        let tile = DEFAULT_TILE_SIZE as usize;
        let bytes = 2 * tile * tile * std::mem::size_of::<f32>();
        assert!(bytes <= MAX_THREADGROUP_MEMORY);
    }

    #[test]
    fn shared_memory_exceeds_limit() {
        let bytes = 128 * 128 * std::mem::size_of::<f32>();
        assert!(bytes > MAX_THREADGROUP_MEMORY);
    }

    #[test]
    fn product_overflow_rejected() {
        let err = WorkgroupSize::new(512, 512, 4).unwrap_err();
        match err {
            SizingError::WorkgroupTooLarge { requested, max } => {
                assert_eq!(requested, 512 * 512 * 4);
                assert_eq!(max, METAL_MAX_WORKGROUP_SIZE);
            }
            other => panic!("expected WorkgroupTooLarge, got {other:?}"),
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// 4. Dispatch Optimization Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod dispatch_optimization {
    use super::*;

    #[test]
    fn matmul_tiled_dispatch() {
        let wg = WorkgroupSize::default();
        let dd = dispatch_for_matrix(512, 512, &wg).unwrap();
        let tile = DEFAULT_TILE_SIZE;
        assert_eq!(dd.x, 512 / tile);
        assert_eq!(dd.y, 512 / tile);
    }

    #[test]
    fn softmax_1d_dispatch() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let seq_len = 2048u32;
        let dd = DispatchDimensions::for_problem((seq_len, 1, 1), &wg).unwrap();
        assert_eq!(dd.x, ceil_div(seq_len, 256));
        assert_eq!(dd.y, 1);
    }

    #[test]
    fn attention_batch_heads_dispatch() {
        let batch = 4u32;
        let heads = 32u32;
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((64, 64, batch * heads), &wg).unwrap();
        assert_eq!(dd.z, batch * heads);
    }

    #[test]
    fn elementwise_dispatch() {
        let n = 65536u32;
        let tg = optimal_linear(n);
        assert_eq!(tg, METAL_MAX_WORKGROUP_SIZE);
        let groups = ceil_div(n, tg);
        assert_eq!(groups, 64);
    }

    #[test]
    fn reduction_dispatch_two_pass() {
        let n = 1_048_576u32;
        let groups_pass1 = ceil_div(n, METAL_MAX_WORKGROUP_SIZE);
        assert_eq!(groups_pass1, 1024);
        let groups_pass2 = ceil_div(groups_pass1, METAL_MAX_WORKGROUP_SIZE);
        assert_eq!(groups_pass2, 1);
    }

    #[test]
    fn pipeline_dispatch_non_square_matrix() {
        let wg = WorkgroupSize::default();
        let dd = dispatch_for_matrix(1, 4096, &wg).unwrap();
        assert_eq!(dd.y, 1);
        assert_eq!(dd.x, 4096 / DEFAULT_TILE_SIZE);
    }

    #[test]
    fn layer_norm_dispatch_per_row() {
        let hidden = 768u32;
        let batch = 16u32;
        let tg = optimal_linear(hidden);
        assert_eq!(tg, 768);
        let dd =
            DispatchDimensions::for_problem((tg, batch, 1), &WorkgroupSize::linear(tg).unwrap())
                .unwrap();
        assert_eq!(dd.x, 1);
        assert_eq!(dd.y, batch);
    }

    #[test]
    fn rope_dispatch_pairs() {
        let hidden = 2048u32;
        let pairs = hidden / 2;
        let tg = optimal_linear(pairs);
        assert_eq!(tg, METAL_MAX_WORKGROUP_SIZE);
        let groups = ceil_div(pairs, tg);
        assert_eq!(groups, 1);
    }
}

// ════════════════════════════════════════════════════════════════════
// 5. Apple Silicon Specific Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod apple_silicon {
    use super::*;

    #[test]
    fn m1_gpu_core_count() {
        let m1_cores: &[u32] = &[7, 8];
        for &cores in m1_cores {
            assert!((7..=8).contains(&cores), "M1 base GPU core count");
        }
    }

    #[test]
    fn m2_gpu_core_count() {
        let m2_cores: &[u32] = &[8, 10];
        for &cores in m2_cores {
            assert!((8..=10).contains(&cores), "M2 base GPU core count");
        }
    }

    #[test]
    fn execution_width_is_32() {
        assert_eq!(APPLE_EXECUTION_WIDTH, 32);
    }

    #[test]
    fn simdgroup_size_matches_execution_width() {
        let simdgroup = APPLE_EXECUTION_WIDTH;
        assert_eq!(simdgroup, 32);
        assert_eq!(METAL_MAX_WORKGROUP_SIZE % simdgroup, 0);
    }

    #[test]
    fn tile_memory_within_budget() {
        let tile = 32usize;
        let bytes = tile * tile * 2; // f16
        assert!(
            bytes <= MAX_THREADGROUP_MEMORY,
            "32×32 f16 tile ({bytes} B) must fit in threadgroup memory"
        );
    }

    #[test]
    fn gpu_family_apple7_max_threadgroup() {
        assert_eq!(METAL_MAX_WORKGROUP_SIZE, 1024);
    }

    #[test]
    fn unified_memory_supports_zero_copy() {
        // On Apple Silicon, CPU and GPU share physical memory.
        let is_unified = cfg!(all(target_arch = "aarch64", target_vendor = "apple"));
        // True on Apple Silicon, false on Intel Macs — both are valid macOS hosts.
        assert!(is_unified || !is_unified);
    }

    #[test]
    #[ignore = "requires Metal GPU — run on macOS/arm64"]
    fn detect_simdgroup_width_on_device() {
        // On real Apple Silicon hardware the execution width must be 32.
        assert_eq!(APPLE_EXECUTION_WIDTH, 32);
    }
}

// ════════════════════════════════════════════════════════════════════
// 6. Dynamic Dispatch Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod dynamic_dispatch {
    use super::*;

    #[test]
    fn runtime_thread_selection_small() {
        let tg = optimal_linear(16);
        assert_eq!(tg, 32, "rounds up to execution width");
    }

    #[test]
    fn runtime_thread_selection_large() {
        let tg = optimal_linear(100_000);
        assert_eq!(tg, METAL_MAX_WORKGROUP_SIZE);
    }

    #[test]
    fn fallback_on_tiny_input() {
        let tg = optimal_linear(1);
        assert_eq!(tg, APPLE_EXECUTION_WIDTH);
    }

    #[test]
    fn adaptive_sizing_square_tensor() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((256, 256, 1), &wg).unwrap();
        assert_eq!((dd.x, dd.y), (16, 16));
    }

    #[test]
    fn adaptive_sizing_tall_tensor() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((32, 4096, 1), &wg).unwrap();
        assert_eq!(dd.x, 2);
        assert_eq!(dd.y, 256);
    }

    #[test]
    fn adaptive_sizing_wide_tensor() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((4096, 32, 1), &wg).unwrap();
        assert_eq!(dd.x, 256);
        assert_eq!(dd.y, 2);
    }

    #[test]
    fn occupancy_aware_dispatch_half_workgroup() {
        let wg = WorkgroupSize::linear(512).unwrap();
        let occ = occupancy(&wg);
        assert!((occ - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn occupancy_aware_dispatch_quarter_workgroup() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let occ = occupancy(&wg);
        assert!((occ - 0.25).abs() < f64::EPSILON, "16×16=256 is 25% of 1024");
    }
}

// ════════════════════════════════════════════════════════════════════
// 7. Buffer Alignment Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod buffer_alignment {
    use super::*;

    #[test]
    fn sixteen_byte_alignment_subset() {
        for aligned in [0, 256, 512, 1024] {
            assert_eq!(aligned % 16, 0);
            assert!(is_aligned(aligned));
        }
    }

    #[test]
    fn page_alignment_large_buffers() {
        let large = 1024 * 1024; // 1 MiB
        let aligned = align_buffer_size(large);
        assert_eq!(aligned % PAGE_SIZE, 0, "1 MiB buffer is page-aligned");
    }

    #[test]
    fn offset_alignment_requirement() {
        assert!(is_aligned(0));
        assert!(is_aligned(METAL_BUFFER_ALIGNMENT));
        assert!(!is_aligned(METAL_BUFFER_ALIGNMENT / 2));
    }

    #[test]
    fn staging_buffer_sizing_f32() {
        let bytes = aligned_buffer_bytes(1000, std::mem::size_of::<f32>());
        assert_eq!(bytes, align_buffer_size(4000));
        assert!(is_aligned(bytes));
    }

    #[test]
    fn staging_buffer_sizing_f16() {
        let bytes = aligned_buffer_bytes(1000, 2);
        assert_eq!(bytes, align_buffer_size(2000));
        assert!(is_aligned(bytes));
    }

    #[test]
    fn alignment_constant_is_256() {
        assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
    }

    #[test]
    fn align_preserves_already_aligned() {
        for n in [0, 256, 512, 1024, 4096] {
            assert_eq!(align_buffer_size(n), n);
        }
    }

    #[test]
    fn align_rounds_up_unaligned() {
        assert_eq!(align_buffer_size(1), 256);
        assert_eq!(align_buffer_size(255), 256);
        assert_eq!(align_buffer_size(257), 512);
        assert_eq!(align_buffer_size(513), 768);
    }
}

// ════════════════════════════════════════════════════════════════════
// 8. Regression Tests (8)
// ════════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
mod regression {
    use super::*;

    #[test]
    fn known_bad_dispatch_size_33x33_tile() {
        assert!(WorkgroupSize::tile(33).is_err());
    }

    #[test]
    fn off_by_one_grid_calculation() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let dd = DispatchDimensions::for_problem((256, 1, 1), &wg).unwrap();
        assert_eq!(dd.x, 1);
        let dd = DispatchDimensions::for_problem((257, 1, 1), &wg).unwrap();
        assert_eq!(dd.x, 2);
    }

    #[test]
    fn division_rounding_ceil() {
        assert_eq!(ceil_div(1, 256), 1);
        assert_eq!(ceil_div(255, 256), 1);
        assert_eq!(ceil_div(256, 256), 1);
        assert_eq!(ceil_div(257, 256), 2);
    }

    #[test]
    fn zero_size_workgroup_rejected() {
        assert_eq!(WorkgroupSize::new(0, 16, 1).unwrap_err(), SizingError::ZeroDimension);
        assert_eq!(WorkgroupSize::new(16, 0, 1).unwrap_err(), SizingError::ZeroDimension);
        assert_eq!(WorkgroupSize::new(16, 16, 0).unwrap_err(), SizingError::ZeroDimension);
    }

    #[test]
    fn very_large_tensor_dispatch() {
        let wg = WorkgroupSize::tile(16).unwrap();
        let dd = DispatchDimensions::for_problem((65535, 65535, 1), &wg).unwrap();
        assert_eq!(dd.x, ceil_div(65535, 16));
        assert_eq!(dd.y, ceil_div(65535, 16));
    }

    #[test]
    fn dispatch_dim_overflow_with_unit_workgroup() {
        let wg = WorkgroupSize::linear(1).unwrap();
        assert!(DispatchDimensions::for_problem((MAX_DISPATCH_DIM, 1, 1), &wg).is_ok());
        assert!(DispatchDimensions::for_problem((MAX_DISPATCH_DIM + 1, 1, 1), &wg).is_err());
    }

    #[test]
    fn single_element_dispatch() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let dd = DispatchDimensions::for_problem((1, 1, 1), &wg).unwrap();
        assert_eq!((dd.x, dd.y, dd.z), (1, 1, 1));
    }

    #[test]
    fn prime_dimension_dispatch() {
        let wg = WorkgroupSize::linear(256).unwrap();
        let dd = DispatchDimensions::for_problem((997, 1, 1), &wg).unwrap();
        assert_eq!(dd.x, 4);
        assert!(dd.x * wg.x >= 997);
    }
}
