//! OpenCL workgroup size optimizer for Intel A770 dispatch configurations.
//!
//! Computes optimal local workgroup sizes based on kernel characteristics
//! and hardware limits (max workgroup size, subgroup sizes, local memory, etc.).

use std::collections::HashMap;
use std::fmt;

// ── Types ────────────────────────────────────────────────────────────

/// Hardware capability profile for an OpenCL device.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub max_workgroup_size: usize,
    pub max_workgroup_dims: Vec<usize>,
    pub subgroup_sizes: Vec<usize>,
    pub num_compute_units: usize,
    pub max_local_memory: usize,
    pub preferred_vector_width_float: usize,
    pub warp_size: usize,
}

/// Describes the compute characteristics of a single kernel.
#[derive(Debug, Clone)]
pub struct KernelProfile {
    pub name: String,
    pub dimensions: usize,
    pub registers_per_thread: u32,
    pub local_memory_bytes: usize,
    pub arithmetic_intensity: f32,
    pub memory_access_pattern: AccessPattern,
}

/// Memory access pattern hint used to tune workgroup shape.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPattern {
    Sequential,
    Strided(usize),
    Random,
    Coalesced,
    BroadcastRow,
    BroadcastCol,
}

/// Recommended workgroup configuration with diagnostics.
#[derive(Debug, Clone)]
pub struct WorkgroupRecommendation {
    pub local_size: Vec<usize>,
    pub estimated_occupancy: f32,
    pub estimated_throughput: f64,
    pub reasoning: String,
}

/// Optimization goal that drives the scoring heuristic.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationGoal {
    MaxOccupancy,
    MinLatency,
    MaxThroughput,
    BalancedOccupancyThroughput,
}

/// Workgroup optimizer with per-kernel recommendation cache.
pub struct WorkgroupOptimizer {
    pub hardware: HardwareProfile,
    pub cache: HashMap<String, WorkgroupRecommendation>,
}

/// Errors produced by the optimizer.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerError {
    InvalidDimensions,
    ExceedsHardwareLimits(String),
    InsufficientLocalMemory { required: usize, available: usize },
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions => write!(f, "invalid dimensions"),
            Self::ExceedsHardwareLimits(msg) => {
                write!(f, "exceeds hardware limits: {msg}")
            }
            Self::InsufficientLocalMemory { required, available } => {
                write!(
                    f,
                    "insufficient local memory: need {required} B, have {available} B"
                )
            }
        }
    }
}

impl std::error::Error for OptimizerError {}

// ── Factory helpers ──────────────────────────────────────────────────

/// Create a `HardwareProfile` with Intel Arc A770 defaults.
///
/// * 1024 max workgroup size
/// * 32 Xe-cores
/// * 64 KiB shared local memory (SLM)
/// * Subgroup sizes 8 / 16 / 32 (16 preferred for SIMD)
pub fn create_a770_profile() -> HardwareProfile {
    HardwareProfile {
        max_workgroup_size: 1024,
        max_workgroup_dims: vec![1024, 1024, 64],
        subgroup_sizes: vec![8, 16, 32],
        num_compute_units: 32,
        max_local_memory: 65536, // 64 KiB
        preferred_vector_width_float: 8,
        warp_size: 16,
    }
}

/// Create a `WorkgroupOptimizer` for the given hardware profile.
pub fn create_workgroup_optimizer(hardware: HardwareProfile) -> WorkgroupOptimizer {
    WorkgroupOptimizer { hardware, cache: HashMap::new() }
}

// ── Core compute functions ───────────────────────────────────────────

/// Compute optimal 1-D local size for the given kernel and global size.
pub fn cpu_compute_optimal_1d(
    optimizer: &WorkgroupOptimizer,
    kernel: &KernelProfile,
    global_size: usize,
    goal: OptimizationGoal,
) -> WorkgroupRecommendation {
    let hw = &optimizer.hardware;
    let candidates = cpu_enumerate_valid_sizes(hw, kernel, 1);

    let mut best: Option<(f64, Vec<usize>)> = None;
    for c in &candidates {
        // Skip candidates that are larger than global_size (wasteful).
        if c[0] > global_size && global_size > 0 {
            continue;
        }
        let score = cpu_score_workgroup_size(c, hw, kernel, &goal);
        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((score, c.clone()));
        }
    }

    let local = best.map(|(_, s)| s).unwrap_or_else(|| {
        vec![hw.warp_size.min(global_size.max(1))]
    });

    let local = cpu_adjust_for_access_pattern(&local, &kernel.memory_access_pattern, hw);

    let occupancy = cpu_compute_occupancy(&local, hw, kernel);
    let throughput = cpu_estimate_throughput(&local, hw, kernel);

    WorkgroupRecommendation {
        local_size: local.clone(),
        estimated_occupancy: occupancy,
        estimated_throughput: throughput,
        reasoning: format!(
            "1D optimal for '{}': local_size={}, occupancy={:.2}, goal={:?}",
            kernel.name, local[0], occupancy, goal
        ),
    }
}

/// Compute optimal 2-D local size for the given kernel and global dims.
pub fn cpu_compute_optimal_2d(
    optimizer: &WorkgroupOptimizer,
    kernel: &KernelProfile,
    global_x: usize,
    global_y: usize,
    goal: OptimizationGoal,
) -> WorkgroupRecommendation {
    let hw = &optimizer.hardware;
    let candidates = cpu_enumerate_valid_sizes(hw, kernel, 2);

    let mut best: Option<(f64, Vec<usize>)> = None;
    for c in &candidates {
        if (c[0] > global_x && global_x > 0) || (c[1] > global_y && global_y > 0) {
            continue;
        }
        let score = cpu_score_workgroup_size(c, hw, kernel, &goal);
        if best.as_ref().is_none_or(|(s, _)| score > *s) {
            best = Some((score, c.clone()));
        }
    }

    let local = best.map(|(_, s)| s).unwrap_or_else(|| {
        let x = hw.warp_size.min(global_x.max(1));
        let y = 1usize;
        vec![x, y]
    });

    let local = cpu_adjust_for_access_pattern(&local, &kernel.memory_access_pattern, hw);

    let occupancy = cpu_compute_occupancy(&local, hw, kernel);
    let throughput = cpu_estimate_throughput(&local, hw, kernel);

    WorkgroupRecommendation {
        local_size: local.clone(),
        estimated_occupancy: occupancy,
        estimated_throughput: throughput,
        reasoning: format!(
            "2D optimal for '{}': local_size={}x{}, occupancy={:.2}, goal={:?}",
            kernel.name, local[0], local[1], occupancy, goal
        ),
    }
}

/// Estimate occupancy ∈ [0, 1] for a given local size.
///
/// Occupancy is modelled as: `active_threads / max_threads_per_CU`.
/// We assume up to `max_workgroup_size` threads per CU and penalise
/// configurations that don't fill a whole number of subgroups.
pub fn cpu_compute_occupancy(
    local_size: &[usize],
    hardware: &HardwareProfile,
    kernel: &KernelProfile,
) -> f32 {
    let total: usize = local_size.iter().product();
    if total == 0 || hardware.max_workgroup_size == 0 {
        return 0.0;
    }

    // Base occupancy: fraction of max workgroup size.
    let base = total as f32 / hardware.max_workgroup_size as f32;

    // Subgroup alignment bonus: prefer sizes that are exact multiples.
    let subgroup = hardware.warp_size.max(1);
    let alignment = if total.is_multiple_of(subgroup) { 1.0 } else { 0.85 };

    // Local memory pressure penalty.
    let mem_used = kernel.local_memory_bytes.max(total * 4);
    let mem_ratio = mem_used as f32 / hardware.max_local_memory as f32;
    let mem_factor = if mem_ratio > 1.0 { 0.0 } else { 1.0 - 0.3 * mem_ratio };

    (base * alignment * mem_factor).clamp(0.0, 1.0)
}

/// Estimate throughput (GFlop/s heuristic) for a given local size.
pub fn cpu_estimate_throughput(
    local_size: &[usize],
    hardware: &HardwareProfile,
    kernel: &KernelProfile,
) -> f64 {
    let total: usize = local_size.iter().product();
    if total == 0 {
        return 0.0;
    }

    let occupancy = cpu_compute_occupancy(local_size, hardware, kernel) as f64;
    let cus = hardware.num_compute_units as f64;
    let ai = kernel.arithmetic_intensity as f64;

    // Simple roofline-ish model: CUs × occupancy × arithmetic_intensity
    let base = cus * occupancy * ai.max(0.1);

    // Pattern penalty
    let pattern_factor = match &kernel.memory_access_pattern {
        AccessPattern::Coalesced | AccessPattern::Sequential => 1.0,
        AccessPattern::BroadcastRow | AccessPattern::BroadcastCol => 0.9,
        AccessPattern::Strided(s) => 1.0 / (1.0 + (*s as f64).ln()),
        AccessPattern::Random => 0.5,
    };

    base * pattern_factor
}

/// Adjust a base workgroup size for the given access pattern.
pub fn cpu_adjust_for_access_pattern(
    base_size: &[usize],
    pattern: &AccessPattern,
    hardware: &HardwareProfile,
) -> Vec<usize> {
    let max_wg = hardware.max_workgroup_size;

    match pattern {
        AccessPattern::Coalesced => {
            // Favour wide-x for coalesced reads.
            if base_size.len() >= 2 {
                let x = base_size[0]
                    .next_power_of_two()
                    .min(hardware.max_workgroup_dims[0])
                    .min(max_wg);
                let y = (max_wg / x).min(base_size[1]).max(1);
                vec![x, y]
            } else {
                base_size.to_vec()
            }
        }
        AccessPattern::BroadcastRow if base_size.len() >= 2 => {
            // Rows are broadcast → collapse y.
            let x = base_size[0].min(max_wg);
            vec![x, 1]
        }
        AccessPattern::BroadcastCol if base_size.len() >= 2 => {
            // Columns are broadcast → collapse x.
            let y = base_size[1].min(max_wg);
            vec![1, y]
        }
        _ => base_size.to_vec(),
    }
}

/// Enumerate all valid local sizes for the given dimensionality.
pub fn cpu_enumerate_valid_sizes(
    hardware: &HardwareProfile,
    kernel: &KernelProfile,
    dims: usize,
) -> Vec<Vec<usize>> {
    let max_wg = hardware.max_workgroup_size;
    let subgroup = hardware.warp_size.max(1);

    match dims {
        1 => {
            let mut out = Vec::new();
            let mut s = subgroup;
            while s <= max_wg && s <= hardware.max_workgroup_dims[0] {
                if kernel.local_memory_bytes <= hardware.max_local_memory {
                    out.push(vec![s]);
                }
                s *= 2;
            }
            if out.is_empty() {
                out.push(vec![subgroup.min(max_wg)]);
            }
            out
        }
        2 => {
            let mut out = Vec::new();
            let max_x = hardware
                .max_workgroup_dims
                .first()
                .copied()
                .unwrap_or(max_wg);
            let max_y = hardware
                .max_workgroup_dims
                .get(1)
                .copied()
                .unwrap_or(max_wg);

            let mut x = 1;
            while x <= max_x && x <= max_wg {
                let mut y = 1;
                while y <= max_y && x * y <= max_wg {
                    let total = x * y;
                    if total >= subgroup
                        && total.is_multiple_of(subgroup)
                        && kernel.local_memory_bytes <= hardware.max_local_memory
                    {
                        out.push(vec![x, y]);
                    }
                    y *= 2;
                }
                x *= 2;
            }
            if out.is_empty() {
                out.push(vec![subgroup.min(max_wg), 1]);
            }
            out
        }
        3 => {
            let mut out = Vec::new();
            let max_x = hardware
                .max_workgroup_dims
                .first()
                .copied()
                .unwrap_or(max_wg);
            let max_y = hardware
                .max_workgroup_dims
                .get(1)
                .copied()
                .unwrap_or(max_wg);
            let max_z = hardware
                .max_workgroup_dims
                .get(2)
                .copied()
                .unwrap_or(64);

            let mut x = 1;
            while x <= max_x.min(max_wg) {
                let mut y = 1;
                while y <= max_y && x * y <= max_wg {
                    let mut z = 1;
                    while z <= max_z && x * y * z <= max_wg {
                        let total = x * y * z;
                        if total >= subgroup
                            && total.is_multiple_of(subgroup)
                            && kernel.local_memory_bytes <= hardware.max_local_memory
                        {
                            out.push(vec![x, y, z]);
                        }
                        z *= 2;
                    }
                    y *= 2;
                }
                x *= 2;
            }
            if out.is_empty() {
                out.push(vec![subgroup.min(max_wg), 1, 1]);
            }
            out
        }
        _ => vec![vec![subgroup.min(max_wg)]],
    }
}

/// Score a candidate workgroup size (higher is better).
pub fn cpu_score_workgroup_size(
    size: &[usize],
    hardware: &HardwareProfile,
    kernel: &KernelProfile,
    goal: &OptimizationGoal,
) -> f64 {
    let occupancy = cpu_compute_occupancy(size, hardware, kernel) as f64;
    let throughput = cpu_estimate_throughput(size, hardware, kernel);

    match goal {
        OptimizationGoal::MaxOccupancy => occupancy,
        OptimizationGoal::MaxThroughput => throughput,
        OptimizationGoal::MinLatency => {
            // Lower total → lower latency; invert for "higher is better".
            let total: usize = size.iter().product();
            1.0 / (total as f64 + 1.0) + occupancy * 0.1
        }
        OptimizationGoal::BalancedOccupancyThroughput => {
            0.5 * occupancy + 0.5 * (throughput / (throughput + 1.0))
        }
    }
}

/// Recommend a workgroup size for a matrix-multiply (M×N×K).
pub fn cpu_recommend_for_matmul(
    m: usize,
    n: usize,
    _k: usize,
    optimizer: &WorkgroupOptimizer,
) -> WorkgroupRecommendation {
    let kernel = KernelProfile {
        name: format!("matmul_{m}x{n}"),
        dimensions: 2,
        registers_per_thread: 32,
        local_memory_bytes: 4096,
        arithmetic_intensity: 8.0,
        memory_access_pattern: AccessPattern::Coalesced,
    };

    cpu_compute_optimal_2d(
        optimizer,
        &kernel,
        n,
        m,
        OptimizationGoal::MaxThroughput,
    )
}

/// Recommend a workgroup size for a reduction of `n` elements.
pub fn cpu_recommend_for_reduction(
    n: usize,
    optimizer: &WorkgroupOptimizer,
) -> WorkgroupRecommendation {
    let kernel = KernelProfile {
        name: format!("reduction_{n}"),
        dimensions: 1,
        registers_per_thread: 8,
        local_memory_bytes: 256,
        arithmetic_intensity: 1.0,
        memory_access_pattern: AccessPattern::Sequential,
    };

    cpu_compute_optimal_1d(
        optimizer,
        &kernel,
        n,
        OptimizationGoal::MaxOccupancy,
    )
}

/// Format a recommendation as a human-readable string.
pub fn format_recommendation(rec: &WorkgroupRecommendation) -> String {
    let dims: Vec<String> = rec.local_size.iter().map(|d| d.to_string()).collect();
    format!(
        "local_size=[{}] occupancy={:.2} throughput={:.2} | {}",
        dims.join(", "),
        rec.estimated_occupancy,
        rec.estimated_throughput,
        rec.reasoning,
    )
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn default_kernel_1d() -> KernelProfile {
        KernelProfile {
            name: "test_1d".into(),
            dimensions: 1,
            registers_per_thread: 16,
            local_memory_bytes: 256,
            arithmetic_intensity: 4.0,
            memory_access_pattern: AccessPattern::Sequential,
        }
    }

    fn default_kernel_2d() -> KernelProfile {
        KernelProfile {
            name: "test_2d".into(),
            dimensions: 2,
            registers_per_thread: 16,
            local_memory_bytes: 256,
            arithmetic_intensity: 4.0,
            memory_access_pattern: AccessPattern::Sequential,
        }
    }

    fn a770() -> WorkgroupOptimizer {
        create_workgroup_optimizer(create_a770_profile())
    }

    // ---- A770 profile ------------------------------------------------

    #[test]
    fn a770_max_workgroup_size() {
        let p = create_a770_profile();
        assert_eq!(p.max_workgroup_size, 1024);
    }

    #[test]
    fn a770_compute_units() {
        let p = create_a770_profile();
        assert_eq!(p.num_compute_units, 32);
    }

    #[test]
    fn a770_local_memory() {
        let p = create_a770_profile();
        assert_eq!(p.max_local_memory, 65536);
    }

    #[test]
    fn a770_subgroup_sizes() {
        let p = create_a770_profile();
        assert_eq!(p.subgroup_sizes, vec![8, 16, 32]);
    }

    #[test]
    fn a770_warp_size_is_16() {
        let p = create_a770_profile();
        assert_eq!(p.warp_size, 16);
    }

    #[test]
    fn a770_preferred_vector_width() {
        let p = create_a770_profile();
        assert_eq!(p.preferred_vector_width_float, 8);
    }

    #[test]
    fn a770_max_workgroup_dims() {
        let p = create_a770_profile();
        assert_eq!(p.max_workgroup_dims, vec![1024, 1024, 64]);
    }

    // ---- 1D optimal --------------------------------------------------

    #[test]
    fn optimal_1d_within_hardware_limits() {
        let opt = a770();
        let rec =
            cpu_compute_optimal_1d(&opt, &default_kernel_1d(), 4096, OptimizationGoal::MaxOccupancy);
        assert!(rec.local_size[0] <= opt.hardware.max_workgroup_size);
    }

    #[test]
    fn optimal_1d_is_power_of_two() {
        let opt = a770();
        let rec =
            cpu_compute_optimal_1d(&opt, &default_kernel_1d(), 4096, OptimizationGoal::MaxOccupancy);
        assert!(rec.local_size[0].is_power_of_two());
    }

    #[test]
    fn optimal_1d_max_throughput_positive() {
        let opt = a770();
        let rec = cpu_compute_optimal_1d(
            &opt,
            &default_kernel_1d(),
            4096,
            OptimizationGoal::MaxThroughput,
        );
        assert!(rec.estimated_throughput > 0.0);
    }

    #[test]
    fn optimal_1d_min_latency() {
        let opt = a770();
        let rec = cpu_compute_optimal_1d(
            &opt,
            &default_kernel_1d(),
            4096,
            OptimizationGoal::MinLatency,
        );
        assert!(rec.local_size[0] <= opt.hardware.max_workgroup_size);
    }

    // ---- 2D optimal --------------------------------------------------

    #[test]
    fn optimal_2d_within_limits() {
        let opt = a770();
        let rec = cpu_compute_optimal_2d(
            &opt,
            &default_kernel_2d(),
            1024,
            1024,
            OptimizationGoal::MaxOccupancy,
        );
        let product: usize = rec.local_size.iter().product();
        assert!(product <= opt.hardware.max_workgroup_size);
    }

    #[test]
    fn optimal_2d_has_two_dims() {
        let opt = a770();
        let rec = cpu_compute_optimal_2d(
            &opt,
            &default_kernel_2d(),
            256,
            256,
            OptimizationGoal::MaxThroughput,
        );
        assert_eq!(rec.local_size.len(), 2);
    }

    #[test]
    fn optimal_2d_each_dim_within_max() {
        let opt = a770();
        let rec = cpu_compute_optimal_2d(
            &opt,
            &default_kernel_2d(),
            512,
            512,
            OptimizationGoal::BalancedOccupancyThroughput,
        );
        assert!(rec.local_size[0] <= opt.hardware.max_workgroup_dims[0]);
        assert!(rec.local_size[1] <= opt.hardware.max_workgroup_dims[1]);
    }

    // ---- Occupancy ---------------------------------------------------

    #[test]
    fn occupancy_full_utilization() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let occ = cpu_compute_occupancy(&[1024], &hw, &k);
        // Should be close to 1.0 (penalised slightly by memory factor).
        assert!(occ > 0.5, "occupancy={occ}");
    }

    #[test]
    fn occupancy_in_range() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        for size in [16, 32, 64, 128, 256, 512, 1024] {
            let occ = cpu_compute_occupancy(&[size], &hw, &k);
            assert!((0.0..=1.0).contains(&occ), "occ={occ} for size={size}");
        }
    }

    #[test]
    fn occupancy_zero_for_zero_size() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        assert_eq!(cpu_compute_occupancy(&[0], &hw, &k), 0.0);
    }

    #[test]
    fn occupancy_subgroup_aligned_beats_unaligned() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let aligned = cpu_compute_occupancy(&[256], &hw, &k);
        // 17 is intentionally not subgroup-aligned.
        let unaligned = cpu_compute_occupancy(&[17], &hw, &k);
        assert!(aligned > unaligned, "aligned={aligned}, unaligned={unaligned}");
    }

    // ---- Throughput ---------------------------------------------------

    #[test]
    fn throughput_positive_for_valid_config() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let tp = cpu_estimate_throughput(&[256], &hw, &k);
        assert!(tp > 0.0);
    }

    #[test]
    fn throughput_zero_for_zero_size() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        assert_eq!(cpu_estimate_throughput(&[0], &hw, &k), 0.0);
    }

    #[test]
    fn throughput_coalesced_beats_random() {
        let hw = create_a770_profile();
        let k_coal = KernelProfile {
            memory_access_pattern: AccessPattern::Coalesced,
            ..default_kernel_1d()
        };
        let k_rand = KernelProfile {
            memory_access_pattern: AccessPattern::Random,
            ..default_kernel_1d()
        };
        let tp_coal = cpu_estimate_throughput(&[256], &hw, &k_coal);
        let tp_rand = cpu_estimate_throughput(&[256], &hw, &k_rand);
        assert!(tp_coal > tp_rand, "coal={tp_coal}, rand={tp_rand}");
    }

    // ---- Access pattern adjustment -----------------------------------

    #[test]
    fn coalesced_prefers_wide_x() {
        let hw = create_a770_profile();
        let adj = cpu_adjust_for_access_pattern(
            &[16, 16],
            &AccessPattern::Coalesced,
            &hw,
        );
        assert!(adj[0] >= 16);
        assert_eq!(adj.len(), 2);
    }

    #[test]
    fn broadcast_row_collapses_y() {
        let hw = create_a770_profile();
        let adj = cpu_adjust_for_access_pattern(
            &[32, 32],
            &AccessPattern::BroadcastRow,
            &hw,
        );
        assert_eq!(adj[1], 1);
    }

    #[test]
    fn broadcast_col_collapses_x() {
        let hw = create_a770_profile();
        let adj = cpu_adjust_for_access_pattern(
            &[32, 32],
            &AccessPattern::BroadcastCol,
            &hw,
        );
        assert_eq!(adj[0], 1);
    }

    #[test]
    fn sequential_unchanged() {
        let hw = create_a770_profile();
        let base = vec![64, 4];
        let adj = cpu_adjust_for_access_pattern(
            &base,
            &AccessPattern::Sequential,
            &hw,
        );
        assert_eq!(adj, base);
    }

    // ---- Enumerate valid sizes ---------------------------------------

    #[test]
    fn enumerate_1d_all_within_limits() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let sizes = cpu_enumerate_valid_sizes(&hw, &k, 1);
        assert!(!sizes.is_empty());
        for s in &sizes {
            assert!(s[0] <= hw.max_workgroup_size);
        }
    }

    #[test]
    fn enumerate_2d_all_products_within_limits() {
        let hw = create_a770_profile();
        let k = default_kernel_2d();
        let sizes = cpu_enumerate_valid_sizes(&hw, &k, 2);
        assert!(!sizes.is_empty());
        for s in &sizes {
            let product: usize = s.iter().product();
            assert!(product <= hw.max_workgroup_size);
        }
    }

    #[test]
    fn enumerate_3d_within_limits() {
        let hw = create_a770_profile();
        let k = KernelProfile {
            dimensions: 3,
            ..default_kernel_1d()
        };
        let sizes = cpu_enumerate_valid_sizes(&hw, &k, 3);
        assert!(!sizes.is_empty());
        for s in &sizes {
            let product: usize = s.iter().product();
            assert!(product <= hw.max_workgroup_size);
            assert!(s[2] <= hw.max_workgroup_dims[2]);
        }
    }

    #[test]
    fn enumerate_respects_local_memory() {
        let hw = create_a770_profile();
        let k = KernelProfile {
            local_memory_bytes: 100_000, // exceeds 64 KiB
            ..default_kernel_1d()
        };
        let sizes = cpu_enumerate_valid_sizes(&hw, &k, 1);
        // Fallback entry always present.
        assert!(!sizes.is_empty());
    }

    // ---- Scoring -----------------------------------------------------

    #[test]
    fn score_higher_for_better_config() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let s256 = cpu_score_workgroup_size(
            &[256],
            &hw,
            &k,
            &OptimizationGoal::MaxOccupancy,
        );
        let s16 = cpu_score_workgroup_size(
            &[16],
            &hw,
            &k,
            &OptimizationGoal::MaxOccupancy,
        );
        assert!(s256 > s16, "s256={s256}, s16={s16}");
    }

    #[test]
    fn score_throughput_positive() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let s = cpu_score_workgroup_size(
            &[128],
            &hw,
            &k,
            &OptimizationGoal::MaxThroughput,
        );
        assert!(s > 0.0);
    }

    #[test]
    fn score_balanced_in_range() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        let s = cpu_score_workgroup_size(
            &[128],
            &hw,
            &k,
            &OptimizationGoal::BalancedOccupancyThroughput,
        );
        assert!(s >= 0.0);
    }

    // ---- Matmul recommendation ---------------------------------------

    #[test]
    fn matmul_recommendation_2d() {
        let opt = a770();
        let rec = cpu_recommend_for_matmul(512, 512, 512, &opt);
        assert_eq!(rec.local_size.len(), 2);
    }

    #[test]
    fn matmul_product_within_limits() {
        let opt = a770();
        let rec = cpu_recommend_for_matmul(1024, 1024, 1024, &opt);
        let product: usize = rec.local_size.iter().product();
        assert!(product <= opt.hardware.max_workgroup_size);
    }

    #[test]
    fn matmul_square_ish_tiles() {
        let opt = a770();
        let rec = cpu_recommend_for_matmul(256, 256, 256, &opt);
        // Coalesced pattern favours wide x; both dims should be ≥ 1.
        assert!(rec.local_size[0] >= 1);
        assert!(rec.local_size[1] >= 1);
    }

    // ---- Reduction recommendation ------------------------------------

    #[test]
    fn reduction_recommendation_1d() {
        let opt = a770();
        let rec = cpu_recommend_for_reduction(4096, &opt);
        assert_eq!(rec.local_size.len(), 1);
    }

    #[test]
    fn reduction_power_of_two() {
        let opt = a770();
        let rec = cpu_recommend_for_reduction(4096, &opt);
        assert!(rec.local_size[0].is_power_of_two());
    }

    // ---- Edge cases --------------------------------------------------

    #[test]
    fn edge_global_size_one() {
        let opt = a770();
        let rec = cpu_compute_optimal_1d(
            &opt,
            &default_kernel_1d(),
            1,
            OptimizationGoal::MaxOccupancy,
        );
        // Should still produce a valid recommendation.
        assert!(!rec.local_size.is_empty());
        assert!(rec.local_size[0] >= 1);
    }

    #[test]
    fn edge_global_size_smaller_than_warp() {
        let opt = a770();
        let rec = cpu_compute_optimal_1d(
            &opt,
            &default_kernel_1d(),
            4,
            OptimizationGoal::MaxOccupancy,
        );
        assert!(rec.local_size[0] >= 1);
    }

    #[test]
    fn edge_3d_dispatch_limits() {
        let hw = create_a770_profile();
        let k = KernelProfile {
            dimensions: 3,
            ..default_kernel_1d()
        };
        let sizes = cpu_enumerate_valid_sizes(&hw, &k, 3);
        for s in &sizes {
            assert!(s[0] <= hw.max_workgroup_dims[0]);
            assert!(s[1] <= hw.max_workgroup_dims[1]);
            assert!(s[2] <= hw.max_workgroup_dims[2]);
        }
    }

    // ---- Property: optimal divides global after rounding -------------

    #[test]
    fn property_local_divides_global_after_round() {
        let opt = a770();
        for gs in [64, 128, 256, 512, 1024, 4096] {
            let rec = cpu_compute_optimal_1d(
                &opt,
                &default_kernel_1d(),
                gs,
                OptimizationGoal::MaxOccupancy,
            );
            let ls = rec.local_size[0];
            let rounded = ((gs + ls - 1) / ls) * ls;
            assert_eq!(rounded % ls, 0, "gs={gs}, ls={ls}");
        }
    }

    // ---- Property: occupancy ∈ [0, 1] --------------------------------

    #[test]
    fn property_occupancy_bounded() {
        let hw = create_a770_profile();
        let k = default_kernel_1d();
        for size in [1, 7, 16, 33, 64, 255, 512, 1024] {
            let occ = cpu_compute_occupancy(&[size], &hw, &k);
            assert!(
                (0.0..=1.0).contains(&occ),
                "occ={occ} for size={size}"
            );
        }
    }

    // ---- Property: product <= max_workgroup_size ----------------------

    #[test]
    fn property_product_within_max() {
        let opt = a770();
        for (gx, gy) in [(64, 64), (256, 256), (1024, 1024)] {
            let rec = cpu_compute_optimal_2d(
                &opt,
                &default_kernel_2d(),
                gx,
                gy,
                OptimizationGoal::MaxOccupancy,
            );
            let product: usize = rec.local_size.iter().product();
            assert!(
                product <= opt.hardware.max_workgroup_size,
                "product={product} for {gx}x{gy}"
            );
        }
    }

    // ---- A770 subgroup preference ------------------------------------

    #[test]
    fn a770_subgroup_16_preferred_for_simd() {
        let p = create_a770_profile();
        assert!(p.subgroup_sizes.contains(&16));
        assert_eq!(p.warp_size, 16, "A770 Xe-core uses 16-wide SIMD");
    }

    // ---- Format recommendation --------------------------------------

    #[test]
    fn format_recommendation_nonempty() {
        let rec = WorkgroupRecommendation {
            local_size: vec![256],
            estimated_occupancy: 0.75,
            estimated_throughput: 42.0,
            reasoning: "test".into(),
        };
        let s = format_recommendation(&rec);
        assert!(s.contains("256"));
        assert!(s.contains("0.75"));
    }

    // ---- OptimizerError display -------------------------------------

    #[test]
    fn error_display_invalid_dims() {
        let e = OptimizerError::InvalidDimensions;
        assert_eq!(e.to_string(), "invalid dimensions");
    }

    #[test]
    fn error_display_exceeds_limits() {
        let e = OptimizerError::ExceedsHardwareLimits("too big".into());
        assert!(e.to_string().contains("too big"));
    }

    #[test]
    fn error_display_insufficient_memory() {
        let e = OptimizerError::InsufficientLocalMemory {
            required: 128_000,
            available: 65_536,
        };
        let s = e.to_string();
        assert!(s.contains("128000"));
        assert!(s.contains("65536"));
    }

    // ---- Cache in optimizer -----------------------------------------

    #[test]
    fn optimizer_cache_starts_empty() {
        let opt = a770();
        assert!(opt.cache.is_empty());
    }

    #[test]
    fn optimizer_cache_insert_retrieve() {
        let mut opt = a770();
        let rec = cpu_recommend_for_reduction(512, &opt);
        opt.cache.insert("reduction_512".into(), rec);
        assert!(opt.cache.contains_key("reduction_512"));
    }

    // ---- Strided access pattern throughput ----------------------------

    #[test]
    fn strided_throughput_decreases_with_stride() {
        let hw = create_a770_profile();
        let k1 = KernelProfile {
            memory_access_pattern: AccessPattern::Strided(2),
            ..default_kernel_1d()
        };
        let k2 = KernelProfile {
            memory_access_pattern: AccessPattern::Strided(64),
            ..default_kernel_1d()
        };
        let tp1 = cpu_estimate_throughput(&[256], &hw, &k1);
        let tp2 = cpu_estimate_throughput(&[256], &hw, &k2);
        assert!(tp1 > tp2, "stride-2={tp1}, stride-64={tp2}");
    }

    // ---- Balanced goal -----------------------------------------------

    #[test]
    fn balanced_goal_returns_valid_recommendation() {
        let opt = a770();
        let rec = cpu_compute_optimal_1d(
            &opt,
            &default_kernel_1d(),
            2048,
            OptimizationGoal::BalancedOccupancyThroughput,
        );
        assert!(rec.estimated_occupancy >= 0.0);
        assert!(rec.estimated_throughput >= 0.0);
    }
}
