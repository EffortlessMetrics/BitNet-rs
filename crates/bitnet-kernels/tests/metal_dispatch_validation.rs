#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal compute dispatch validation tests for Apple Silicon.
//!
//! Validates dispatch configuration logic: threadgroup sizing, grid dimension
//! calculations, occupancy estimation, Apple Silicon chip profiles, and
//! workload balancing across threadgroups.
//!
//! These are pure-logic tests (no GPU required) that validate the dispatch
//! planner maths used by the Metal backend.

// ── Mock types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct DispatchConfig {
    grid_size: (u32, u32, u32),
    threadgroup_size: (u32, u32, u32),
    total_threads: u64,
}

impl DispatchConfig {
    fn total_threadgroups(&self) -> u64 {
        self.grid_size.0 as u64 * self.grid_size.1 as u64 * self.grid_size.2 as u64
    }

    fn threads_per_threadgroup(&self) -> u64 {
        self.threadgroup_size.0 as u64
            * self.threadgroup_size.1 as u64
            * self.threadgroup_size.2 as u64
    }

    fn covers(&self, work_items: u64) -> bool {
        self.total_threadgroups() * self.threads_per_threadgroup() >= work_items
    }
}

#[derive(Debug, Clone)]
struct GpuProfile {
    name: &'static str,
    max_threadgroup_size: u32,
    max_threads_per_threadgroup: u32,
    simd_width: u32,
    gpu_cores: u32,
    max_threadgroups_per_grid: (u32, u32, u32),
}

impl GpuProfile {
    fn m1() -> Self {
        Self {
            name: "Apple M1",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 8,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m1_pro() -> Self {
        Self {
            name: "Apple M1 Pro",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 16,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m1_max() -> Self {
        Self {
            name: "Apple M1 Max",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 32,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m1_ultra() -> Self {
        Self {
            name: "Apple M1 Ultra",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 64,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m2() -> Self {
        Self {
            name: "Apple M2",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 10,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m2_pro() -> Self {
        Self {
            name: "Apple M2 Pro",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 19,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m2_max() -> Self {
        Self {
            name: "Apple M2 Max",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 38,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m2_ultra() -> Self {
        Self {
            name: "Apple M2 Ultra",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 76,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m3() -> Self {
        Self {
            name: "Apple M3",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 10,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m3_pro() -> Self {
        Self {
            name: "Apple M3 Pro",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 18,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m3_max() -> Self {
        Self {
            name: "Apple M3 Max",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 40,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m4() -> Self {
        Self {
            name: "Apple M4",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 10,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m4_pro() -> Self {
        Self {
            name: "Apple M4 Pro",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 20,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn m4_max() -> Self {
        Self {
            name: "Apple M4 Max",
            max_threadgroup_size: 1024,
            max_threads_per_threadgroup: 1024,
            simd_width: 32,
            gpu_cores: 40,
            max_threadgroups_per_grid: (65535, 65535, 65535),
        }
    }

    fn all_profiles() -> Vec<Self> {
        vec![
            Self::m1(),
            Self::m1_pro(),
            Self::m1_max(),
            Self::m1_ultra(),
            Self::m2(),
            Self::m2_pro(),
            Self::m2_max(),
            Self::m2_ultra(),
            Self::m3(),
            Self::m3_pro(),
            Self::m3_max(),
            Self::m4(),
            Self::m4_pro(),
            Self::m4_max(),
        ]
    }
}

// ── Dispatch helpers ────────────────────────────────────────────────────────

/// Choose a 1-D threadgroup size that is a multiple of SIMD width, capped at
/// `max_tg` and no larger than `work_items`.
fn select_threadgroup_size_1d(work_items: u64, simd_width: u32, max_tg: u32) -> u32 {
    let mut tg = simd_width;
    while tg * 2 <= max_tg && (tg as u64) * 2 <= work_items {
        tg *= 2;
    }
    tg
}

/// Build a 1-D dispatch configuration.
fn dispatch_1d(work_items: u64, profile: &GpuProfile) -> DispatchConfig {
    let tg = select_threadgroup_size_1d(
        work_items,
        profile.simd_width,
        profile.max_threads_per_threadgroup,
    );
    let groups = (work_items).div_ceil(tg as u64) as u32;
    DispatchConfig {
        grid_size: (groups, 1, 1),
        threadgroup_size: (tg, 1, 1),
        total_threads: groups as u64 * tg as u64,
    }
}

/// Build a 2-D dispatch configuration for a (rows × cols) workload.
fn dispatch_2d(rows: u32, cols: u32, profile: &GpuProfile) -> DispatchConfig {
    let max_tg = profile.max_threads_per_threadgroup;
    // Prefer square-ish threadgroups; each dimension a power of 2.
    let mut tg_x: u32 = profile.simd_width.min(cols);
    let mut tg_y: u32 = 1;
    while tg_x * tg_y * 2 <= max_tg && tg_y * 2 <= rows {
        tg_y *= 2;
    }
    // Try expanding x if room left.
    while tg_x * 2 * tg_y <= max_tg && tg_x * 2 <= cols {
        tg_x *= 2;
    }
    let grid_x = (cols as u64).div_ceil(tg_x as u64) as u32;
    let grid_y = (rows as u64).div_ceil(tg_y as u64) as u32;
    DispatchConfig {
        grid_size: (grid_x, grid_y, 1),
        threadgroup_size: (tg_x, tg_y, 1),
        total_threads: grid_x as u64 * grid_y as u64 * tg_x as u64 * tg_y as u64,
    }
}

/// Build a 3-D dispatch configuration for a (x × y × z) workload.
fn dispatch_3d(dim_x: u32, dim_y: u32, dim_z: u32, profile: &GpuProfile) -> DispatchConfig {
    let max_tg = profile.max_threads_per_threadgroup;
    let mut tg_x = profile.simd_width.min(dim_x);
    let mut tg_y: u32 = 1;
    let mut tg_z: u32 = 1;
    while tg_x * tg_y * tg_z * 2 <= max_tg && tg_y * 2 <= dim_y {
        tg_y *= 2;
    }
    while tg_x * tg_y * tg_z * 2 <= max_tg && tg_z * 2 <= dim_z {
        tg_z *= 2;
    }
    let grid_x = (dim_x as u64).div_ceil(tg_x as u64) as u32;
    let grid_y = (dim_y as u64).div_ceil(tg_y as u64) as u32;
    let grid_z = (dim_z as u64).div_ceil(tg_z as u64) as u32;
    DispatchConfig {
        grid_size: (grid_x, grid_y, grid_z),
        threadgroup_size: (tg_x, tg_y, tg_z),
        total_threads: grid_x as u64
            * grid_y as u64
            * grid_z as u64
            * tg_x as u64
            * tg_y as u64
            * tg_z as u64,
    }
}

/// Estimate occupancy as the ratio of active threads to theoretical max.
fn estimate_occupancy(config: &DispatchConfig, profile: &GpuProfile) -> f64 {
    let threads_per_tg = config.threads_per_threadgroup();
    let max = profile.max_threads_per_threadgroup as f64;
    let simd = profile.simd_width as f64;
    // Occupancy: fraction of max threads that are actually utilised, weighted
    // by SIMD lane utilisation of the last warp.
    let simd_util = if threads_per_tg.is_multiple_of(profile.simd_width as u64) {
        1.0
    } else {
        (threads_per_tg % profile.simd_width as u64) as f64 / simd
    };
    let tg_util = threads_per_tg as f64 / max;
    tg_util * simd_util
}

/// Compute the waste ratio: fraction of dispatched threads that exceed `work_items`.
fn waste_ratio(config: &DispatchConfig, work_items: u64) -> f64 {
    let total = config.total_threadgroups() * config.threads_per_threadgroup();
    if total == 0 {
        return 1.0;
    }
    (total - work_items.min(total)) as f64 / total as f64
}

/// Balance a 1-D workload across `num_chunks` equal-ish parts, returning
/// `(chunk_size, remainder)`.
fn balance_workload(total: u64, num_chunks: u32) -> (u64, u64) {
    let chunk = total / num_chunks as u64;
    let rem = total % num_chunks as u64;
    (chunk, rem)
}

/// Recommend the number of threadgroups that keeps all GPU cores busy.
fn recommended_threadgroups(profile: &GpuProfile) -> u32 {
    // Heuristic: 4 threadgroups per GPU core for latency hiding.
    profile.gpu_cores * 4
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: threadgroup_sizing
// ═══════════════════════════════════════════════════════════════════════════

mod threadgroup_sizing {
    use super::*;

    #[test]
    fn selects_simd_width_for_tiny_workload() {
        let tg = select_threadgroup_size_1d(1, 32, 1024);
        assert_eq!(tg, 32, "minimum threadgroup should equal SIMD width");
    }

    #[test]
    fn selects_simd_width_when_work_equals_simd() {
        assert_eq!(select_threadgroup_size_1d(32, 32, 1024), 32);
    }

    #[test]
    fn doubles_up_to_64() {
        assert_eq!(select_threadgroup_size_1d(64, 32, 1024), 64);
    }

    #[test]
    fn doubles_up_to_128() {
        assert_eq!(select_threadgroup_size_1d(200, 32, 1024), 128);
    }

    #[test]
    fn doubles_up_to_256() {
        assert_eq!(select_threadgroup_size_1d(300, 32, 1024), 256);
    }

    #[test]
    fn doubles_up_to_512() {
        assert_eq!(select_threadgroup_size_1d(600, 32, 1024), 512);
    }

    #[test]
    fn caps_at_1024_for_large_workloads() {
        assert_eq!(select_threadgroup_size_1d(100_000, 32, 1024), 1024);
    }

    #[test]
    fn respects_max_tg_limit_256() {
        assert_eq!(select_threadgroup_size_1d(100_000, 32, 256), 256);
    }

    #[test]
    fn respects_max_tg_limit_64() {
        assert_eq!(select_threadgroup_size_1d(100_000, 32, 64), 64);
    }

    #[test]
    fn result_always_power_of_two() {
        for work in [1, 7, 33, 100, 1000, 50_000, 1_000_000u64] {
            let tg = select_threadgroup_size_1d(work, 32, 1024);
            assert!(tg.is_power_of_two(), "tg {tg} for work {work} is not power of 2");
        }
    }

    #[test]
    fn result_always_multiple_of_simd_width() {
        for simd in [32u32] {
            for work in [1, 50, 500, 5000, 50_000u64] {
                let tg = select_threadgroup_size_1d(work, simd, 1024);
                assert_eq!(tg % simd, 0, "tg {tg} not multiple of SIMD {simd}");
            }
        }
    }

    #[test]
    fn never_exceeds_max_threadgroup_size() {
        for max in [32, 64, 128, 256, 512, 1024u32] {
            let tg = select_threadgroup_size_1d(u64::MAX, 32, max);
            assert!(tg <= max, "tg {tg} exceeded max {max}");
        }
    }

    #[test]
    fn threadgroup_32_for_small_tensor() {
        let cfg = dispatch_1d(16, &GpuProfile::m1());
        assert_eq!(cfg.threadgroup_size.0, 32);
    }

    #[test]
    fn threadgroup_256_for_medium_tensor() {
        let cfg = dispatch_1d(4096, &GpuProfile::m1());
        assert!(cfg.threadgroup_size.0 >= 256);
    }

    #[test]
    fn threadgroup_1024_for_large_tensor() {
        let cfg = dispatch_1d(1_000_000, &GpuProfile::m1());
        assert_eq!(cfg.threadgroup_size.0, 1024);
    }

    #[test]
    fn all_valid_sizes_are_standard() {
        let valid = [32, 64, 128, 256, 512, 1024];
        for n in [1, 31, 32, 33, 63, 64, 65, 127, 255, 513, 1023, 2048, 100_000u64] {
            let tg = select_threadgroup_size_1d(n, 32, 1024);
            assert!(valid.contains(&tg), "tg {tg} not in standard set");
        }
    }

    #[test]
    fn monotonically_non_decreasing_with_work() {
        let mut prev = 0u32;
        for exp in 0..20 {
            let work = 1u64 << exp;
            let tg = select_threadgroup_size_1d(work, 32, 1024);
            assert!(tg >= prev, "tg decreased from {prev} to {tg} at work {work}");
            prev = tg;
        }
    }

    #[test]
    fn exact_power_of_two_workloads() {
        for exp in 5..=10 {
            let work = 1u64 << exp;
            let tg = select_threadgroup_size_1d(work, 32, 1024);
            assert_eq!(tg as u64, work, "exact P2 workload {work} => tg {tg}");
        }
    }

    #[test]
    fn typical_bitnet_hidden_dim_2048() {
        let tg = select_threadgroup_size_1d(2048, 32, 1024);
        assert_eq!(tg, 1024);
    }

    #[test]
    fn typical_bitnet_hidden_dim_4096() {
        let tg = select_threadgroup_size_1d(4096, 32, 1024);
        assert_eq!(tg, 1024);
    }

    #[test]
    fn vocab_size_128256() {
        let tg = select_threadgroup_size_1d(128_256, 32, 1024);
        assert_eq!(tg, 1024);
    }

    #[test]
    fn threadgroup_size_never_zero() {
        for w in [1, 2, 15, 16, 31, 100_000u64] {
            let tg = select_threadgroup_size_1d(w, 32, 1024);
            assert!(tg > 0, "tg is zero for work {w}");
        }
    }

    #[test]
    fn max_tg_32_clamps_immediately() {
        let tg = select_threadgroup_size_1d(10_000, 32, 32);
        assert_eq!(tg, 32);
    }

    #[test]
    fn hidden_dim_8192_large_model() {
        let tg = select_threadgroup_size_1d(8192, 32, 1024);
        assert_eq!(tg, 1024);
    }

    #[test]
    fn intermediate_dim_5461() {
        // 2048 * 8/3 ≈ 5461 (SwiGLU intermediate)
        let tg = select_threadgroup_size_1d(5461, 32, 1024);
        assert_eq!(tg, 1024);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: dispatch_dimensions
// ═══════════════════════════════════════════════════════════════════════════

mod dispatch_dimensions {
    use super::*;

    // -- 1-D dispatch --

    #[test]
    fn dispatch_1d_single_element() {
        let cfg = dispatch_1d(1, &GpuProfile::m1());
        assert!(cfg.covers(1));
        assert_eq!(cfg.grid_size.0, 1);
    }

    #[test]
    fn dispatch_1d_exact_one_threadgroup() {
        let cfg = dispatch_1d(1024, &GpuProfile::m1());
        assert!(cfg.covers(1024));
        assert_eq!(cfg.grid_size.0, 1);
        assert_eq!(cfg.threadgroup_size.0, 1024);
    }

    #[test]
    fn dispatch_1d_two_threadgroups() {
        let cfg = dispatch_1d(1025, &GpuProfile::m1());
        assert!(cfg.covers(1025));
        assert!(cfg.grid_size.0 >= 2);
    }

    #[test]
    fn dispatch_1d_large_tensor() {
        let work = 1_048_576u64; // 1M
        let cfg = dispatch_1d(work, &GpuProfile::m1());
        assert!(cfg.covers(work));
        assert_eq!(cfg.threadgroup_size.0, 1024);
        assert_eq!(cfg.grid_size.0, 1024);
    }

    #[test]
    fn dispatch_1d_odd_size() {
        let cfg = dispatch_1d(997, &GpuProfile::m1());
        assert!(cfg.covers(997));
    }

    #[test]
    fn dispatch_1d_prime_size() {
        let cfg = dispatch_1d(7919, &GpuProfile::m1());
        assert!(cfg.covers(7919));
    }

    #[test]
    fn dispatch_1d_waste_under_50_pct_for_moderate_sizes() {
        for work in [100, 500, 1000, 4096, 10_000u64] {
            let cfg = dispatch_1d(work, &GpuProfile::m1());
            let w = waste_ratio(&cfg, work);
            assert!(w < 0.5, "waste {w:.2} too high for work {work}");
        }
    }

    #[test]
    fn dispatch_1d_waste_under_1_pct_for_large() {
        let work = 1_000_000u64;
        let cfg = dispatch_1d(work, &GpuProfile::m1());
        let w = waste_ratio(&cfg, work);
        assert!(w < 0.01, "waste {w:.4} too high for work {work}");
    }

    // -- 2-D dispatch --

    #[test]
    fn dispatch_2d_square_matrix() {
        let cfg = dispatch_2d(256, 256, &GpuProfile::m1());
        assert!(cfg.covers(256 * 256));
        assert!(cfg.threadgroup_size.1 > 1, "should use 2-D threadgroups");
    }

    #[test]
    fn dispatch_2d_tall_matrix() {
        let cfg = dispatch_2d(4096, 32, &GpuProfile::m1());
        assert!(cfg.covers(4096 * 32));
    }

    #[test]
    fn dispatch_2d_wide_matrix() {
        let cfg = dispatch_2d(32, 4096, &GpuProfile::m1());
        assert!(cfg.covers(32 * 4096));
    }

    #[test]
    fn dispatch_2d_single_row() {
        let cfg = dispatch_2d(1, 1024, &GpuProfile::m1());
        assert!(cfg.covers(1024));
        assert_eq!(cfg.threadgroup_size.1, 1);
    }

    #[test]
    fn dispatch_2d_single_col() {
        let cfg = dispatch_2d(1024, 1, &GpuProfile::m1());
        assert!(cfg.covers(1024));
    }

    #[test]
    fn dispatch_2d_respects_max_threads() {
        let cfg = dispatch_2d(1024, 1024, &GpuProfile::m1());
        let tg_total = cfg.threadgroup_size.0 * cfg.threadgroup_size.1;
        assert!(tg_total <= 1024, "threadgroup {tg_total} exceeds limit");
    }

    #[test]
    fn dispatch_2d_typical_attention_shape() {
        // batch=1, seq_len=512, hidden=2048
        let cfg = dispatch_2d(512, 2048, &GpuProfile::m3());
        assert!(cfg.covers(512 * 2048));
    }

    #[test]
    fn dispatch_2d_gemm_like() {
        let cfg = dispatch_2d(2048, 2048, &GpuProfile::m2());
        assert!(cfg.covers(2048 * 2048));
        let tg = cfg.threadgroup_size.0 * cfg.threadgroup_size.1;
        assert!(tg >= 64, "GEMM dispatch should use ≥64 threads per TG");
    }

    // -- 3-D dispatch --

    #[test]
    fn dispatch_3d_small_cube() {
        let cfg = dispatch_3d(8, 8, 8, &GpuProfile::m1());
        assert!(cfg.covers(512));
    }

    #[test]
    fn dispatch_3d_batch_seq_hidden() {
        // (batch=4, seq=128, hidden=2048)
        let cfg = dispatch_3d(4, 128, 2048, &GpuProfile::m3());
        assert!(cfg.covers(4 * 128 * 2048));
    }

    #[test]
    fn dispatch_3d_respects_max_threads() {
        let cfg = dispatch_3d(64, 64, 64, &GpuProfile::m1());
        let tg = cfg.threadgroup_size.0 * cfg.threadgroup_size.1 * cfg.threadgroup_size.2;
        assert!(tg <= 1024);
    }

    #[test]
    fn dispatch_3d_degenerate_z1() {
        let cfg = dispatch_3d(256, 256, 1, &GpuProfile::m1());
        assert_eq!(cfg.threadgroup_size.2, 1);
        assert_eq!(cfg.grid_size.2, 1);
    }

    #[test]
    fn dispatch_3d_degenerate_y1_z1() {
        let cfg = dispatch_3d(10000, 1, 1, &GpuProfile::m1());
        assert_eq!(cfg.threadgroup_size.1, 1);
        assert_eq!(cfg.threadgroup_size.2, 1);
    }

    #[test]
    fn dispatch_covers_all_elements_1d() {
        for n in [1, 31, 32, 33, 127, 1023, 1024, 1025, 65537, 1_000_000u64] {
            let cfg = dispatch_1d(n, &GpuProfile::m1());
            assert!(cfg.covers(n), "1-D dispatch does not cover {n} elements");
        }
    }

    #[test]
    fn dispatch_covers_all_elements_2d() {
        for (r, c) in [(1, 1), (3, 7), (64, 64), (1024, 512), (1, 100_000)] {
            let cfg = dispatch_2d(r, c, &GpuProfile::m1());
            assert!(cfg.covers(r as u64 * c as u64), "2-D miss for ({r},{c})");
        }
    }

    #[test]
    fn grid_size_within_limits() {
        let profile = GpuProfile::m1();
        // 65535 × 1024 = ~67M — fits in one grid dimension.
        let cfg = dispatch_1d(60_000_000, &profile);
        assert!(cfg.grid_size.0 <= profile.max_threadgroups_per_grid.0);
    }

    #[test]
    fn grid_size_2d_within_limits() {
        let profile = GpuProfile::m1();
        // Each dimension needs ≤65535 threadgroups.
        let cfg = dispatch_2d(4096, 4096, &profile);
        assert!(cfg.grid_size.0 <= profile.max_threadgroups_per_grid.0);
        assert!(cfg.grid_size.1 <= profile.max_threadgroups_per_grid.1);
    }

    #[test]
    fn dispatch_1d_power_of_two_perfect_fit() {
        for exp in 5..=17 {
            let work = 1u64 << exp;
            let cfg = dispatch_1d(work, &GpuProfile::m1());
            assert_eq!(cfg.total_threads, work, "imperfect fit for 2^{exp}");
        }
    }

    #[test]
    fn dispatch_2d_non_square_aspect_ratio() {
        let cfg = dispatch_2d(2, 65536, &GpuProfile::m1());
        assert!(cfg.covers(2 * 65536));
    }

    #[test]
    fn dispatch_3d_single_element() {
        let cfg = dispatch_3d(1, 1, 1, &GpuProfile::m1());
        assert!(cfg.covers(1));
        assert_eq!(cfg.total_threadgroups(), 1);
    }

    #[test]
    fn dispatch_2d_1x1() {
        let cfg = dispatch_2d(1, 1, &GpuProfile::m1());
        assert!(cfg.covers(1));
    }

    #[test]
    fn dispatch_1d_yields_yz_trivial() {
        let cfg = dispatch_1d(4096, &GpuProfile::m1());
        assert_eq!(cfg.grid_size.1, 1);
        assert_eq!(cfg.grid_size.2, 1);
        assert_eq!(cfg.threadgroup_size.1, 1);
        assert_eq!(cfg.threadgroup_size.2, 1);
    }

    #[test]
    fn dispatch_3d_large_z() {
        let cfg = dispatch_3d(2, 2, 8192, &GpuProfile::m3());
        assert!(cfg.covers(2 * 2 * 8192));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: occupancy_estimation
// ═══════════════════════════════════════════════════════════════════════════

mod occupancy_estimation {
    use super::*;

    #[test]
    fn full_occupancy_at_max_tg() {
        let profile = GpuProfile::m1();
        let cfg = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (1024, 1, 1),
            total_threads: 1024,
        };
        let occ = estimate_occupancy(&cfg, &profile);
        assert!((occ - 1.0).abs() < 1e-9, "full TG should be 100% occupancy, got {occ}");
    }

    #[test]
    fn half_occupancy_at_512() {
        let profile = GpuProfile::m1();
        let cfg = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (512, 1, 1),
            total_threads: 512,
        };
        let occ = estimate_occupancy(&cfg, &profile);
        assert!((occ - 0.5).abs() < 1e-9, "512/1024 should be 50%, got {occ}");
    }

    #[test]
    fn quarter_occupancy_at_256() {
        let profile = GpuProfile::m1();
        let cfg = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (256, 1, 1),
            total_threads: 256,
        };
        let occ = estimate_occupancy(&cfg, &profile);
        assert!((occ - 0.25).abs() < 1e-9, "got {occ}");
    }

    #[test]
    fn occupancy_positive_for_single_simd() {
        let profile = GpuProfile::m1();
        let cfg = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (32, 1, 1),
            total_threads: 32,
        };
        let occ = estimate_occupancy(&cfg, &profile);
        assert!(occ > 0.0);
    }

    #[test]
    fn occupancy_penalises_partial_simd() {
        let profile = GpuProfile::m1();
        let full = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (64, 1, 1),
            total_threads: 64,
        };
        let partial = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (48, 1, 1),
            total_threads: 48,
        };
        let occ_full = estimate_occupancy(&full, &profile);
        let occ_partial = estimate_occupancy(&partial, &profile);
        assert!(
            occ_full > occ_partial,
            "full SIMD ({occ_full}) should beat partial ({occ_partial})"
        );
    }

    #[test]
    fn occupancy_never_exceeds_1() {
        let profile = GpuProfile::m1();
        for tg in [32, 64, 128, 256, 512, 1024] {
            let cfg = DispatchConfig {
                grid_size: (100, 1, 1),
                threadgroup_size: (tg, 1, 1),
                total_threads: 100 * tg as u64,
            };
            let occ = estimate_occupancy(&cfg, &profile);
            assert!(occ <= 1.0 + 1e-9, "occupancy {occ} > 1.0 for tg {tg}");
        }
    }

    #[test]
    fn occupancy_consistent_across_profiles_same_tg() {
        // All M-series share the same max_threads_per_threadgroup (1024) and
        // simd_width (32), so occupancy should be identical for the same config.
        let cfg = DispatchConfig {
            grid_size: (10, 1, 1),
            threadgroup_size: (256, 1, 1),
            total_threads: 2560,
        };
        let vals: Vec<f64> =
            GpuProfile::all_profiles().iter().map(|p| estimate_occupancy(&cfg, p)).collect();
        for v in &vals {
            assert!((v - vals[0]).abs() < 1e-9, "occupancy diverged across profiles");
        }
    }

    #[test]
    fn higher_tg_gives_higher_occupancy() {
        let profile = GpuProfile::m1();
        let mut prev = 0.0f64;
        for tg in [32, 64, 128, 256, 512, 1024u32] {
            let cfg = DispatchConfig {
                grid_size: (1, 1, 1),
                threadgroup_size: (tg, 1, 1),
                total_threads: tg as u64,
            };
            let occ = estimate_occupancy(&cfg, &profile);
            assert!(occ >= prev, "occ decreased from {prev} to {occ} at tg {tg}");
            prev = occ;
        }
    }

    #[test]
    fn occupancy_2d_threadgroup() {
        let profile = GpuProfile::m1();
        let cfg = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (32, 32, 1), // 1024 total
            total_threads: 1024,
        };
        let occ = estimate_occupancy(&cfg, &profile);
        assert!((occ - 1.0).abs() < 1e-9);
    }

    #[test]
    fn occupancy_3d_threadgroup() {
        let profile = GpuProfile::m1();
        let cfg = DispatchConfig {
            grid_size: (1, 1, 1),
            threadgroup_size: (8, 8, 16), // 1024 total
            total_threads: 1024,
        };
        let occ = estimate_occupancy(&cfg, &profile);
        assert!((occ - 1.0).abs() < 1e-9);
    }

    #[test]
    fn waste_zero_for_exact_fit() {
        let cfg = DispatchConfig {
            grid_size: (4, 1, 1),
            threadgroup_size: (256, 1, 1),
            total_threads: 1024,
        };
        assert!((waste_ratio(&cfg, 1024) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn waste_positive_for_non_exact() {
        let cfg = dispatch_1d(1000, &GpuProfile::m1());
        let w = waste_ratio(&cfg, 1000);
        assert!(w > 0.0, "non-exact should have positive waste");
    }

    #[test]
    fn waste_bounded_by_one_threadgroup() {
        for work in [1, 33, 100, 500, 1023u64] {
            let cfg = dispatch_1d(work, &GpuProfile::m1());
            let wasted_threads = cfg.total_threads - work;
            let tg = cfg.threads_per_threadgroup();
            assert!(
                wasted_threads < tg,
                "waste {wasted_threads} >= threadgroup {tg} for work {work}"
            );
        }
    }

    #[test]
    fn recommended_threadgroups_m1() {
        let r = recommended_threadgroups(&GpuProfile::m1());
        assert_eq!(r, 32); // 8 cores × 4
    }

    #[test]
    fn recommended_threadgroups_m1_ultra() {
        let r = recommended_threadgroups(&GpuProfile::m1_ultra());
        assert_eq!(r, 256); // 64 cores × 4
    }

    #[test]
    fn recommended_threadgroups_scales_with_cores() {
        let r_m1 = recommended_threadgroups(&GpuProfile::m1());
        let r_m1_max = recommended_threadgroups(&GpuProfile::m1_max());
        assert!(r_m1_max > r_m1);
    }

    #[test]
    fn occupancy_monotone_for_aligned_tg_sizes() {
        let profile = GpuProfile::m3();
        let aligned = [32, 64, 128, 256, 512, 1024u32];
        let occs: Vec<f64> = aligned
            .iter()
            .map(|&tg| {
                let cfg = DispatchConfig {
                    grid_size: (1, 1, 1),
                    threadgroup_size: (tg, 1, 1),
                    total_threads: tg as u64,
                };
                estimate_occupancy(&cfg, &profile)
            })
            .collect();
        for i in 1..occs.len() {
            assert!(occs[i] >= occs[i - 1], "occupancy not monotone at index {i}");
        }
    }

    #[test]
    fn waste_ratio_zero_work_items() {
        let cfg = dispatch_1d(32, &GpuProfile::m1());
        // 0 work items means everything is waste, but covers returns true.
        let w = waste_ratio(&cfg, 0);
        assert!((w - 1.0).abs() < 1e-9);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: apple_silicon_profiles
// ═══════════════════════════════════════════════════════════════════════════

mod apple_silicon_profiles {
    use super::*;

    #[test]
    fn m1_base_profile() {
        let p = GpuProfile::m1();
        assert_eq!(p.gpu_cores, 8);
        assert_eq!(p.simd_width, 32);
        assert_eq!(p.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn m1_pro_has_more_cores_than_m1() {
        assert!(GpuProfile::m1_pro().gpu_cores > GpuProfile::m1().gpu_cores);
    }

    #[test]
    fn m1_max_has_more_cores_than_m1_pro() {
        assert!(GpuProfile::m1_max().gpu_cores > GpuProfile::m1_pro().gpu_cores);
    }

    #[test]
    fn m1_ultra_has_more_cores_than_m1_max() {
        assert!(GpuProfile::m1_ultra().gpu_cores > GpuProfile::m1_max().gpu_cores);
    }

    #[test]
    fn m2_has_more_cores_than_m1() {
        assert!(GpuProfile::m2().gpu_cores >= GpuProfile::m1().gpu_cores);
    }

    #[test]
    fn m2_pro_has_more_cores_than_m2() {
        assert!(GpuProfile::m2_pro().gpu_cores > GpuProfile::m2().gpu_cores);
    }

    #[test]
    fn m2_max_has_more_cores_than_m2_pro() {
        assert!(GpuProfile::m2_max().gpu_cores > GpuProfile::m2_pro().gpu_cores);
    }

    #[test]
    fn m2_ultra_has_more_cores_than_m2_max() {
        assert!(GpuProfile::m2_ultra().gpu_cores > GpuProfile::m2_max().gpu_cores);
    }

    #[test]
    fn m3_profile() {
        let p = GpuProfile::m3();
        assert_eq!(p.gpu_cores, 10);
        assert_eq!(p.simd_width, 32);
    }

    #[test]
    fn m3_pro_has_more_cores_than_m3() {
        assert!(GpuProfile::m3_pro().gpu_cores > GpuProfile::m3().gpu_cores);
    }

    #[test]
    fn m3_max_has_more_cores_than_m3_pro() {
        assert!(GpuProfile::m3_max().gpu_cores > GpuProfile::m3_pro().gpu_cores);
    }

    #[test]
    fn m4_profile() {
        let p = GpuProfile::m4();
        assert_eq!(p.gpu_cores, 10);
        assert_eq!(p.simd_width, 32);
    }

    #[test]
    fn m4_pro_has_more_cores_than_m4() {
        assert!(GpuProfile::m4_pro().gpu_cores > GpuProfile::m4().gpu_cores);
    }

    #[test]
    fn m4_max_has_more_cores_than_m4_pro() {
        assert!(GpuProfile::m4_max().gpu_cores > GpuProfile::m4_pro().gpu_cores);
    }

    #[test]
    fn all_profiles_share_simd_width_32() {
        for p in GpuProfile::all_profiles() {
            assert_eq!(p.simd_width, 32, "{} has wrong SIMD width", p.name);
        }
    }

    #[test]
    fn all_profiles_share_max_tg_1024() {
        for p in GpuProfile::all_profiles() {
            assert_eq!(p.max_threads_per_threadgroup, 1024, "{}", p.name);
        }
    }

    #[test]
    fn all_profiles_have_65535_grid_limits() {
        for p in GpuProfile::all_profiles() {
            assert_eq!(p.max_threadgroups_per_grid, (65535, 65535, 65535), "{}", p.name);
        }
    }

    #[test]
    fn all_profiles_have_nonzero_cores() {
        for p in GpuProfile::all_profiles() {
            assert!(p.gpu_cores > 0, "{} has 0 cores", p.name);
        }
    }

    #[test]
    fn dispatch_1d_consistent_across_profiles() {
        // Same workload should produce the same threadgroup_size on all profiles
        // (they share limits) but grid_size is identical too because
        // select_threadgroup_size_1d only uses simd_width and max.
        let work = 10_000u64;
        let cfgs: Vec<_> =
            GpuProfile::all_profiles().iter().map(|p| dispatch_1d(work, p)).collect();
        for cfg in &cfgs {
            assert_eq!(cfg.threadgroup_size, cfgs[0].threadgroup_size);
            assert_eq!(cfg.grid_size, cfgs[0].grid_size);
        }
    }

    #[test]
    fn recommended_tg_m1_family_scaling() {
        let r_m1 = recommended_threadgroups(&GpuProfile::m1());
        let r_pro = recommended_threadgroups(&GpuProfile::m1_pro());
        let r_max = recommended_threadgroups(&GpuProfile::m1_max());
        let r_ultra = recommended_threadgroups(&GpuProfile::m1_ultra());
        assert!(r_m1 < r_pro);
        assert!(r_pro < r_max);
        assert!(r_max < r_ultra);
    }

    #[test]
    fn recommended_tg_m2_family_scaling() {
        let r_m2 = recommended_threadgroups(&GpuProfile::m2());
        let r_pro = recommended_threadgroups(&GpuProfile::m2_pro());
        let r_max = recommended_threadgroups(&GpuProfile::m2_max());
        let r_ultra = recommended_threadgroups(&GpuProfile::m2_ultra());
        assert!(r_m2 < r_pro);
        assert!(r_pro < r_max);
        assert!(r_max < r_ultra);
    }

    #[test]
    fn recommended_tg_m3_family_scaling() {
        let r_m3 = recommended_threadgroups(&GpuProfile::m3());
        let r_pro = recommended_threadgroups(&GpuProfile::m3_pro());
        let r_max = recommended_threadgroups(&GpuProfile::m3_max());
        assert!(r_m3 < r_pro);
        assert!(r_pro < r_max);
    }

    #[test]
    fn recommended_tg_m4_family_scaling() {
        let r_m4 = recommended_threadgroups(&GpuProfile::m4());
        let r_pro = recommended_threadgroups(&GpuProfile::m4_pro());
        let r_max = recommended_threadgroups(&GpuProfile::m4_max());
        assert!(r_m4 < r_pro);
        assert!(r_pro < r_max);
    }

    #[test]
    fn profile_names_unique() {
        let profiles = GpuProfile::all_profiles();
        let names: Vec<_> = profiles.iter().map(|p| p.name).collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "duplicate profile names");
    }

    #[test]
    fn profile_count_is_14() {
        assert_eq!(GpuProfile::all_profiles().len(), 14);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Module: workload_balancing
// ═══════════════════════════════════════════════════════════════════════════

mod workload_balancing {
    use super::*;

    #[test]
    fn balance_exact_division() {
        let (chunk, rem) = balance_workload(1024, 4);
        assert_eq!(chunk, 256);
        assert_eq!(rem, 0);
    }

    #[test]
    fn balance_with_remainder() {
        let (chunk, rem) = balance_workload(1000, 3);
        assert_eq!(chunk, 333);
        assert_eq!(rem, 1);
        assert_eq!(chunk * 3 + rem, 1000);
    }

    #[test]
    fn balance_single_chunk() {
        let (chunk, rem) = balance_workload(100, 1);
        assert_eq!(chunk, 100);
        assert_eq!(rem, 0);
    }

    #[test]
    fn balance_more_chunks_than_work() {
        let (chunk, rem) = balance_workload(3, 8);
        assert_eq!(chunk, 0);
        assert_eq!(rem, 3);
    }

    #[test]
    fn balance_preserves_total() {
        for (total, chunks) in [(1_000_000, 7), (12345, 13), (1, 1), (99, 100)] {
            let (c, r) = balance_workload(total, chunks);
            assert_eq!(c * chunks as u64 + r, total, "total mismatch for ({total},{chunks})");
        }
    }

    #[test]
    fn balance_remainder_less_than_chunks() {
        for (total, chunks) in [(100, 7), (1024, 3), (999, 10), (1, 5)] {
            let (_, rem) = balance_workload(total, chunks);
            assert!(rem < chunks as u64, "rem {rem} >= chunks {chunks}");
        }
    }

    #[test]
    fn uniform_dispatch_across_m1_cores() {
        let profile = GpuProfile::m1();
        let work = 1_000_000u64;
        let cfg = dispatch_1d(work, &profile);
        let tgs = cfg.total_threadgroups();
        // Every core should get at least one threadgroup.
        assert!(tgs >= profile.gpu_cores as u64, "too few threadgroups for GPU cores");
    }

    #[test]
    fn uniform_dispatch_across_m1_ultra_cores() {
        let profile = GpuProfile::m1_ultra();
        let work = 10_000_000u64;
        let cfg = dispatch_1d(work, &profile);
        let tgs = cfg.total_threadgroups();
        assert!(tgs >= profile.gpu_cores as u64);
    }

    #[test]
    fn uneven_workload_small_remainder() {
        // 1025 elements with TG=1024 → 2 threadgroups, 1023 threads wasted in TG #2.
        let cfg = dispatch_1d(1025, &GpuProfile::m1());
        let w = waste_ratio(&cfg, 1025);
        assert!(w < 0.5, "waste {w:.2} too high");
    }

    #[test]
    fn waste_decreases_with_size() {
        let profile = GpuProfile::m1();
        let w_small = waste_ratio(&dispatch_1d(33, &profile), 33);
        let w_large = waste_ratio(&dispatch_1d(1_000_000, &profile), 1_000_000);
        assert!(w_large < w_small, "large workload should waste less");
    }

    #[test]
    fn dispatch_covers_bitnet_2b_matmul() {
        // 2048 × 2048 = 4M elements — typical BitNet-2B hidden×hidden matmul
        let work = 2048u64 * 2048;
        let cfg = dispatch_1d(work, &GpuProfile::m3());
        assert!(cfg.covers(work));
        assert_eq!(cfg.threadgroup_size.0, 1024);
    }

    #[test]
    fn dispatch_covers_bitnet_2b_layernorm() {
        // LayerNorm over hidden_dim=2048
        let cfg = dispatch_1d(2048, &GpuProfile::m1());
        assert!(cfg.covers(2048));
    }

    #[test]
    fn dispatch_covers_bitnet_2b_vocab_projection() {
        // seq_len=1 × vocab=128256
        let cfg = dispatch_1d(128_256, &GpuProfile::m4());
        assert!(cfg.covers(128_256));
    }

    #[test]
    fn dispatch_2d_bitnet_qkv_projection() {
        // (seq_len=512, 3*hidden=6144) for Q/K/V combined
        let cfg = dispatch_2d(512, 6144, &GpuProfile::m3_max());
        assert!(cfg.covers(512 * 6144));
    }

    #[test]
    fn dispatch_3d_batched_attention() {
        // (batch=4, heads=16, seq_len=256)
        let cfg = dispatch_3d(4, 16, 256, &GpuProfile::m2_pro());
        assert!(cfg.covers(4 * 16 * 256));
    }

    #[test]
    fn balance_across_all_profiles_for_2b_model() {
        let work = 2_000_000_000u64; // 2B params
        for profile in GpuProfile::all_profiles() {
            let rec = recommended_threadgroups(&profile);
            let (chunk, rem) = balance_workload(work, rec);
            assert_eq!(chunk * rec as u64 + rem, work, "total mismatch on {}", profile.name);
        }
    }

    #[test]
    fn threadgroup_count_exceeds_recommended_for_large_work() {
        let profile = GpuProfile::m1();
        let work = 10_000_000u64;
        let cfg = dispatch_1d(work, &profile);
        let rec = recommended_threadgroups(&profile);
        assert!(cfg.total_threadgroups() >= rec as u64, "dispatch should fully utilise GPU");
    }

    #[test]
    fn small_workload_may_underutilise_cores() {
        let profile = GpuProfile::m1_ultra(); // 64 cores
        let cfg = dispatch_1d(32, &profile);
        // Only 1 threadgroup for 32 elements — that's okay for small work.
        assert_eq!(cfg.total_threadgroups(), 1);
    }

    #[test]
    fn waste_is_zero_for_power_of_two_work() {
        for exp in 5..=20 {
            let work = 1u64 << exp;
            let cfg = dispatch_1d(work, &GpuProfile::m1());
            let w = waste_ratio(&cfg, work);
            assert!(w.abs() < 1e-12, "P2 workload 2^{exp} = {work} has non-zero waste {w}");
        }
    }

    #[test]
    fn stress_many_sizes_all_covered() {
        let profile = GpuProfile::m3();
        for size in (1..=500).chain([1023, 1024, 1025, 4096, 65536, 1_000_000].iter().copied()) {
            let cfg = dispatch_1d(size, &profile);
            assert!(cfg.covers(size), "not covered: {size}");
        }
    }

    #[test]
    fn dispatch_config_total_threadgroups_matches_grid() {
        let cfg = DispatchConfig {
            grid_size: (3, 5, 7),
            threadgroup_size: (32, 4, 2),
            total_threads: 3 * 5 * 7 * 32 * 4 * 2,
        };
        assert_eq!(cfg.total_threadgroups(), 105);
        assert_eq!(cfg.threads_per_threadgroup(), 256);
    }

    #[test]
    fn balance_large_model_across_many_chunks() {
        let (chunk, rem) = balance_workload(7_000_000_000, 128);
        assert_eq!(chunk * 128 + rem, 7_000_000_000);
        assert!(rem < 128);
    }
}
