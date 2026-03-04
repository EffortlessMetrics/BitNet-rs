//! Workgroup size optimization for Intel Arc A770 OpenCL kernels.
//!
//! Provides compile-time–safe, runtime-queryable helpers for selecting optimal
//! workgroup dimensions, estimating GPU occupancy, planning shared local memory
//! (SLM) allocation, and aligning global dispatch sizes.  All implementations
//! are **CPU reference code** — no OpenCL runtime (`opencl3`) is required.
//!
//! # Hardware target
//!
//! Intel Arc A770 (Xe-HPG):
//! - 32 Xe-cores, each running up to 8 hardware threads
//! - Subgroup (SIMD) widths: 8, 16, or 32
//! - Max workgroup size: 1024 work-items
//! - 64 KiB shared local memory (SLM) per Xe-core
//! - 128 GRF registers per thread (each 32 bytes → 4 KiB)

use std::fmt;

// ───────────────────────────────────────────────────────────────────────────
// WorkgroupConfig
// ───────────────────────────────────────────────────────────────────────────

/// Three-dimensional workgroup dimensions `(local_x, local_y, local_z)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupConfig {
    pub local_x: usize,
    pub local_y: usize,
    pub local_z: usize,
}

impl WorkgroupConfig {
    /// Create a new workgroup configuration.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    pub fn new(local_x: usize, local_y: usize, local_z: usize) -> Self {
        assert!(local_x > 0 && local_y > 0 && local_z > 0, "workgroup dimensions must be > 0");
        Self { local_x, local_y, local_z }
    }

    /// One-dimensional workgroup.
    pub fn new_1d(local_x: usize) -> Self {
        Self::new(local_x, 1, 1)
    }

    /// Two-dimensional workgroup.
    pub fn new_2d(local_x: usize, local_y: usize) -> Self {
        Self::new(local_x, local_y, 1)
    }

    /// Total number of work-items in the workgroup.
    pub fn total_size(&self) -> usize {
        self.local_x * self.local_y * self.local_z
    }

    /// Number of subgroups (wavefronts) in this workgroup.
    pub fn subgroups(&self, subgroup_size: usize) -> usize {
        assert!(subgroup_size > 0, "subgroup size must be > 0");
        self.total_size().div_ceil(subgroup_size)
    }

    /// Whether total size is a multiple of the given subgroup size.
    pub fn is_subgroup_aligned(&self, subgroup_size: usize) -> bool {
        subgroup_size > 0 && self.total_size().is_multiple_of(subgroup_size)
    }

    /// Return dimensions as a `[usize; 3]` tuple.
    pub fn as_array(&self) -> [usize; 3] {
        [self.local_x, self.local_y, self.local_z]
    }
}

impl fmt::Display for WorkgroupConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.local_z == 1 && self.local_y == 1 {
            write!(f, "({},)", self.local_x)
        } else if self.local_z == 1 {
            write!(f, "({}, {})", self.local_x, self.local_y)
        } else {
            write!(f, "({}, {}, {})", self.local_x, self.local_y, self.local_z)
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// KernelConstraints
// ───────────────────────────────────────────────────────────────────────────

/// Constraints imposed by a specific kernel on workgroup selection.
#[derive(Debug, Clone)]
pub struct KernelConstraints {
    /// Minimum total workgroup size (0 = no minimum).
    pub min_workgroup_size: usize,
    /// Maximum total workgroup size (0 = use device max).
    pub max_workgroup_size: usize,
    /// Required shared local memory in bytes per workgroup.
    pub required_local_memory_bytes: usize,
    /// Whether the kernel uses `barrier()` (occupancy cost).
    pub uses_barriers: bool,
    /// Required subgroup size (`None` = any).
    pub required_subgroup_size: Option<usize>,
    /// Estimated register usage per work-item.
    pub estimated_registers_per_item: usize,
}

impl Default for KernelConstraints {
    fn default() -> Self {
        Self {
            min_workgroup_size: 1,
            max_workgroup_size: 0,
            required_local_memory_bytes: 0,
            uses_barriers: false,
            required_subgroup_size: None,
            estimated_registers_per_item: 32,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// A770WorkgroupProfile
// ───────────────────────────────────────────────────────────────────────────

/// Intel Arc A770 hardware profile.
#[derive(Debug, Clone)]
pub struct A770WorkgroupProfile {
    /// Maximum work-items per workgroup.
    pub max_workgroup_size: usize,
    /// Supported subgroup (SIMD) sizes.
    pub subgroup_sizes: Vec<usize>,
    /// Preferred subgroup size.
    pub preferred_subgroup_size: usize,
    /// Number of Xe-cores.
    pub xe_cores: usize,
    /// Hardware threads per Xe-core.
    pub threads_per_core: usize,
    /// Shared local memory per Xe-core in bytes.
    pub slm_bytes_per_core: usize,
    /// GRF registers per hardware thread.
    pub registers_per_thread: usize,
    /// Bytes per GRF register.
    pub bytes_per_register: usize,
}

impl Default for A770WorkgroupProfile {
    fn default() -> Self {
        Self {
            max_workgroup_size: 1024,
            subgroup_sizes: vec![8, 16, 32],
            preferred_subgroup_size: 16,
            xe_cores: 32,
            threads_per_core: 8,
            slm_bytes_per_core: 65536, // 64 KiB
            registers_per_thread: 128,
            bytes_per_register: 32,
        }
    }
}

impl A770WorkgroupProfile {
    /// Maximum subgroups (wavefronts) schedulable per Xe-core.
    ///
    /// Each hardware thread executes one subgroup, so this equals
    /// `threads_per_core`.
    pub fn max_wavefronts_per_core(&self) -> usize {
        self.threads_per_core
    }

    /// Maximum wavefronts across the entire GPU.
    pub fn max_wavefronts_total(&self) -> usize {
        self.max_wavefronts_per_core() * self.xe_cores
    }

    /// Check if a subgroup size is supported.
    pub fn is_subgroup_size_supported(&self, size: usize) -> bool {
        self.subgroup_sizes.contains(&size)
    }

    /// Total SLM across all Xe-cores.
    pub fn total_slm_bytes(&self) -> usize {
        self.slm_bytes_per_core * self.xe_cores
    }
}

// ───────────────────────────────────────────────────────────────────────────
// RegisterPressure
// ───────────────────────────────────────────────────────────────────────────

/// Estimates register usage and its impact on occupancy.
#[derive(Debug, Clone)]
pub struct RegisterPressure {
    /// Registers used per work-item.
    pub registers_per_item: usize,
    /// Device registers per hardware thread.
    pub registers_per_thread: usize,
    /// Subgroup size for the current dispatch.
    pub subgroup_size: usize,
}

impl RegisterPressure {
    pub fn new(registers_per_item: usize, profile: &A770WorkgroupProfile) -> Self {
        Self {
            registers_per_item,
            registers_per_thread: profile.registers_per_thread,
            subgroup_size: profile.preferred_subgroup_size,
        }
    }

    /// GRF registers consumed by this kernel.
    ///
    /// On Xe-HPG each GRF register is SIMD-wide (serves all lanes in the
    /// subgroup), so total GRF usage equals `registers_per_item`.
    pub fn registers_used(&self) -> usize {
        self.registers_per_item
    }

    /// Maximum concurrent subgroups limited by register file.
    ///
    /// Xe-HPG runs one subgroup per hardware thread.  The kernel can
    /// launch if its GRF usage fits the thread's register file.
    pub fn max_subgroups_per_thread(&self) -> usize {
        if self.registers_per_item == 0 {
            return 1;
        }
        if self.registers_used() > self.registers_per_thread { 0 } else { 1 }
    }

    /// Fraction of the register file consumed by the kernel (0.0–1.0).
    pub fn register_utilization(&self) -> f64 {
        if self.registers_per_thread == 0 {
            return 1.0;
        }
        let used = self.registers_used().min(self.registers_per_thread);
        used as f64 / self.registers_per_thread as f64
    }

    /// Whether the kernel can launch at all (registers fit).
    pub fn can_launch(&self) -> bool {
        self.max_subgroups_per_thread() > 0
    }
}

// ───────────────────────────────────────────────────────────────────────────
// SharedMemoryPlanner
// ───────────────────────────────────────────────────────────────────────────

/// Plans shared local memory (SLM) allocation with occupancy awareness.
#[derive(Debug, Clone)]
pub struct SharedMemoryPlanner {
    /// Total SLM per Xe-core in bytes.
    pub slm_per_core: usize,
    /// Maximum concurrent workgroups per core.
    pub max_workgroups_per_core: usize,
}

impl SharedMemoryPlanner {
    pub fn new(profile: &A770WorkgroupProfile) -> Self {
        Self {
            slm_per_core: profile.slm_bytes_per_core,
            // On Xe-HPG the scheduler can run multiple workgroups on one
            // core if SLM allows.  A practical cap is threads_per_core
            // (each workgroup needs at least one thread).
            max_workgroups_per_core: profile.threads_per_core,
        }
    }

    /// Maximum SLM a single workgroup can request.
    pub fn max_slm_per_workgroup(&self) -> usize {
        self.slm_per_core
    }

    /// Whether the requested amount fits in one core's SLM.
    pub fn fits(&self, bytes: usize) -> bool {
        bytes <= self.slm_per_core
    }

    /// How many workgroups can coexist on one core given their SLM usage.
    pub fn concurrent_workgroups(&self, slm_per_workgroup: usize) -> usize {
        if slm_per_workgroup == 0 {
            return self.max_workgroups_per_core;
        }
        (self.slm_per_core / slm_per_workgroup).min(self.max_workgroups_per_core)
    }

    /// Recommend an SLM budget that keeps at least `target_wgs` workgroups
    /// on one core.  Returns `None` if the target is impossible.
    pub fn budget_for_occupancy(&self, target_wgs: usize) -> Option<usize> {
        if target_wgs == 0 || target_wgs > self.max_workgroups_per_core {
            return None;
        }
        Some(self.slm_per_core / target_wgs)
    }
}

// ───────────────────────────────────────────────────────────────────────────
// OccupancyCalculator
// ───────────────────────────────────────────────────────────────────────────

/// Estimates GPU occupancy for a given workgroup configuration.
#[derive(Debug, Clone)]
pub struct OccupancyCalculator {
    profile: A770WorkgroupProfile,
}

impl OccupancyCalculator {
    pub fn new(profile: A770WorkgroupProfile) -> Self {
        Self { profile }
    }

    /// Create a calculator with default A770 profile.
    pub fn a770() -> Self {
        Self::new(A770WorkgroupProfile::default())
    }

    /// Calculate occupancy for a workgroup + constraints pair.
    ///
    /// Returns `None` if the configuration is infeasible (e.g. workgroup
    /// exceeds device max, registers don't fit, SLM doesn't fit).
    pub fn calculate(
        &self,
        wg: &WorkgroupConfig,
        constraints: &KernelConstraints,
    ) -> Option<WorkgroupStats> {
        let total_size = wg.total_size();

        // ---- feasibility checks ----
        let effective_max = if constraints.max_workgroup_size > 0 {
            constraints.max_workgroup_size.min(self.profile.max_workgroup_size)
        } else {
            self.profile.max_workgroup_size
        };
        if total_size > effective_max || total_size < constraints.min_workgroup_size {
            return None;
        }

        let sg = constraints.required_subgroup_size.unwrap_or(self.profile.preferred_subgroup_size);
        if !self.profile.is_subgroup_size_supported(sg) {
            return None;
        }

        // Registers
        let rp = RegisterPressure {
            registers_per_item: constraints.estimated_registers_per_item,
            registers_per_thread: self.profile.registers_per_thread,
            subgroup_size: sg,
        };
        if !rp.can_launch() {
            return None;
        }

        // SLM
        let planner = SharedMemoryPlanner::new(&self.profile);
        if !planner.fits(constraints.required_local_memory_bytes) {
            return None;
        }

        // ---- occupancy ----
        let subgroups_per_wg = total_size.div_ceil(sg);
        let max_wf = self.profile.max_wavefronts_per_core();

        // Workgroups per core limited by thread slots.
        let wgs_by_threads = if subgroups_per_wg == 0 { 0 } else { max_wf / subgroups_per_wg };

        // Workgroups per core limited by SLM.
        let wgs_by_slm = planner.concurrent_workgroups(constraints.required_local_memory_bytes);

        let wgs_per_core = wgs_by_threads.min(wgs_by_slm).max(1).min(max_wf);

        let active_wavefronts = (wgs_per_core * subgroups_per_wg).min(max_wf);
        let occupancy = active_wavefronts as f64 / max_wf as f64;

        // Subgroup-alignment efficiency.
        let wasted = if total_size.is_multiple_of(sg) { 0 } else { sg - (total_size % sg) };
        let efficiency = (total_size as f64) / ((total_size + wasted) as f64);

        Some(WorkgroupStats {
            workgroup_size: total_size,
            subgroup_size: sg,
            subgroups_per_workgroup: subgroups_per_wg,
            active_wavefronts_per_core: active_wavefronts,
            max_wavefronts_per_core: max_wf,
            occupancy,
            efficiency,
            wasted_items: wasted,
            workgroups_per_core: wgs_per_core,
            register_utilization: rp.register_utilization(),
            slm_per_workgroup: constraints.required_local_memory_bytes,
        })
    }
}

// ───────────────────────────────────────────────────────────────────────────
// WorkgroupStats
// ───────────────────────────────────────────────────────────────────────────

/// Occupancy and efficiency metrics for a workgroup configuration.
#[derive(Debug, Clone)]
pub struct WorkgroupStats {
    /// Total work-items per workgroup.
    pub workgroup_size: usize,
    /// Subgroup (SIMD) width used.
    pub subgroup_size: usize,
    /// Number of subgroups in the workgroup.
    pub subgroups_per_workgroup: usize,
    /// Active wavefronts per Xe-core.
    pub active_wavefronts_per_core: usize,
    /// Maximum wavefronts the core supports.
    pub max_wavefronts_per_core: usize,
    /// Occupancy ratio (0.0–1.0).
    pub occupancy: f64,
    /// Subgroup-alignment efficiency (0.0–1.0).
    pub efficiency: f64,
    /// Work-items wasted due to subgroup padding.
    pub wasted_items: usize,
    /// Concurrent workgroups per Xe-core.
    pub workgroups_per_core: usize,
    /// Fraction of register file used per subgroup (0.0–1.0).
    pub register_utilization: f64,
    /// SLM consumed per workgroup in bytes.
    pub slm_per_workgroup: usize,
}

impl fmt::Display for WorkgroupStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WG={} SG={} occ={:.1}% eff={:.1}% wasted={} wgs/core={}",
            self.workgroup_size,
            self.subgroup_size,
            self.occupancy * 100.0,
            self.efficiency * 100.0,
            self.wasted_items,
            self.workgroups_per_core,
        )
    }
}

// ───────────────────────────────────────────────────────────────────────────
// GlobalSizeAligner
// ───────────────────────────────────────────────────────────────────────────

/// Aligns global work sizes to workgroup multiples so there are no leftover
/// work-items.
#[derive(Debug, Clone)]
pub struct GlobalSizeAligner;

impl GlobalSizeAligner {
    /// Align a 1-D global size to be a multiple of `wg.local_x`.
    pub fn align_1d(problem_size: usize, wg: &WorkgroupConfig) -> usize {
        round_up(problem_size.max(1), wg.local_x)
    }

    /// Align a 2-D global size to workgroup multiples.
    pub fn align_2d(rows: usize, cols: usize, wg: &WorkgroupConfig) -> (usize, usize) {
        (round_up(rows.max(1), wg.local_y), round_up(cols.max(1), wg.local_x))
    }

    /// Align a 3-D global size to workgroup multiples.
    pub fn align_3d(x: usize, y: usize, z: usize, wg: &WorkgroupConfig) -> (usize, usize, usize) {
        (
            round_up(x.max(1), wg.local_x),
            round_up(y.max(1), wg.local_y),
            round_up(z.max(1), wg.local_z),
        )
    }

    /// Total number of workgroups for an aligned 1-D dispatch.
    pub fn workgroup_count_1d(global: usize, wg: &WorkgroupConfig) -> usize {
        global / wg.local_x
    }

    /// Total number of workgroups for an aligned 2-D dispatch.
    pub fn workgroup_count_2d(
        global_rows: usize,
        global_cols: usize,
        wg: &WorkgroupConfig,
    ) -> usize {
        (global_rows / wg.local_y) * (global_cols / wg.local_x)
    }

    /// Wasted work-items in an aligned 1-D dispatch.
    pub fn wasted_1d(problem_size: usize, wg: &WorkgroupConfig) -> usize {
        let aligned = Self::align_1d(problem_size, wg);
        aligned - problem_size
    }

    /// Dispatch efficiency: `useful / total`.
    pub fn efficiency_1d(problem_size: usize, wg: &WorkgroupConfig) -> f64 {
        let aligned = Self::align_1d(problem_size, wg);
        problem_size as f64 / aligned as f64
    }

    /// Dispatch efficiency for a 2-D problem.
    pub fn efficiency_2d(rows: usize, cols: usize, wg: &WorkgroupConfig) -> f64 {
        let (ar, ac) = Self::align_2d(rows, cols, wg);
        (rows * cols) as f64 / (ar * ac) as f64
    }
}

// ───────────────────────────────────────────────────────────────────────────
// WorkgroupOptimizer
// ───────────────────────────────────────────────────────────────────────────

/// Selects the optimal workgroup size for a given kernel and problem.
#[derive(Debug, Clone)]
pub struct WorkgroupOptimizer {
    profile: A770WorkgroupProfile,
    calculator: OccupancyCalculator,
}

impl WorkgroupOptimizer {
    pub fn new(profile: A770WorkgroupProfile) -> Self {
        let calculator = OccupancyCalculator::new(profile.clone());
        Self { profile, calculator }
    }

    /// Create an optimizer with default A770 profile.
    pub fn a770() -> Self {
        Self::new(A770WorkgroupProfile::default())
    }

    /// Select the best 1-D workgroup for a linear problem.
    pub fn optimize_1d(
        &self,
        problem_size: usize,
        constraints: &KernelConstraints,
    ) -> Option<(WorkgroupConfig, WorkgroupStats)> {
        let sg = constraints.required_subgroup_size.unwrap_or(self.profile.preferred_subgroup_size);
        self.search_1d(problem_size, sg, constraints)
    }

    /// Select the best 2-D workgroup for a matrix problem.
    pub fn optimize_2d(
        &self,
        rows: usize,
        cols: usize,
        constraints: &KernelConstraints,
    ) -> Option<(WorkgroupConfig, WorkgroupStats)> {
        let sg = constraints.required_subgroup_size.unwrap_or(self.profile.preferred_subgroup_size);
        self.search_2d(rows, cols, sg, constraints)
    }

    /// Enumerate all feasible 1-D configs with their stats, sorted by
    /// occupancy descending.
    pub fn enumerate_1d(
        &self,
        constraints: &KernelConstraints,
    ) -> Vec<(WorkgroupConfig, WorkgroupStats)> {
        let sg = constraints.required_subgroup_size.unwrap_or(self.profile.preferred_subgroup_size);
        let effective_max = if constraints.max_workgroup_size > 0 {
            constraints.max_workgroup_size.min(self.profile.max_workgroup_size)
        } else {
            self.profile.max_workgroup_size
        };

        let mut results = Vec::new();
        let mut size = sg;
        while size <= effective_max {
            if size >= constraints.min_workgroup_size {
                let wg = WorkgroupConfig::new_1d(size);
                if let Some(stats) = self.calculator.calculate(&wg, constraints) {
                    results.push((wg, stats));
                }
            }
            size += sg;
        }
        results.sort_by(|a, b| {
            b.1.occupancy.partial_cmp(&a.1.occupancy).unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    // ── internal search helpers ──────────────────────────────────────────

    fn search_1d(
        &self,
        problem_size: usize,
        sg: usize,
        constraints: &KernelConstraints,
    ) -> Option<(WorkgroupConfig, WorkgroupStats)> {
        let effective_max = if constraints.max_workgroup_size > 0 {
            constraints.max_workgroup_size.min(self.profile.max_workgroup_size)
        } else {
            self.profile.max_workgroup_size
        };

        let mut best: Option<(WorkgroupConfig, WorkgroupStats, f64)> = None;
        let mut size = sg;
        while size <= effective_max {
            if size < constraints.min_workgroup_size {
                size += sg;
                continue;
            }
            let wg = WorkgroupConfig::new_1d(size);
            if let Some(stats) = self.calculator.calculate(&wg, constraints) {
                let aligned = GlobalSizeAligner::align_1d(problem_size.max(1), &wg);
                let dispatch_eff = problem_size.max(1) as f64 / aligned as f64;
                // Composite score: occupancy + dispatch efficiency + subgroup
                // alignment.
                let score = stats.occupancy * 0.5 + dispatch_eff * 0.3 + stats.efficiency * 0.2;
                if best.as_ref().is_none_or(|(_, _, s)| score > *s) {
                    best = Some((wg, stats, score));
                }
            }
            size += sg;
        }
        best.map(|(wg, stats, _)| (wg, stats))
    }

    fn search_2d(
        &self,
        rows: usize,
        cols: usize,
        sg: usize,
        constraints: &KernelConstraints,
    ) -> Option<(WorkgroupConfig, WorkgroupStats)> {
        let effective_max = if constraints.max_workgroup_size > 0 {
            constraints.max_workgroup_size.min(self.profile.max_workgroup_size)
        } else {
            self.profile.max_workgroup_size
        };

        let mut best: Option<(WorkgroupConfig, WorkgroupStats, f64)> = None;

        // x must be multiple of subgroup size; y varies.
        let mut lx = sg;
        while lx <= effective_max {
            let mut ly = 1usize;
            while lx * ly <= effective_max {
                if lx * ly >= constraints.min_workgroup_size {
                    let wg = WorkgroupConfig::new_2d(lx, ly);
                    if let Some(stats) = self.calculator.calculate(&wg, constraints) {
                        let eff = GlobalSizeAligner::efficiency_2d(rows.max(1), cols.max(1), &wg);
                        let score = stats.occupancy * 0.5 + eff * 0.3 + stats.efficiency * 0.2;
                        if best.as_ref().is_none_or(|(_, _, s)| score > *s) {
                            best = Some((wg, stats, score));
                        }
                    }
                }
                ly += 1;
            }
            lx += sg;
        }
        best.map(|(wg, stats, _)| (wg, stats))
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers (private)
// ───────────────────────────────────────────────────────────────────────────

/// Round `v` up to the next multiple of `m`.
#[inline]
fn round_up(v: usize, m: usize) -> usize {
    if m == 0 {
        return v;
    }
    let r = v % m;
    if r == 0 { v } else { v + m - r }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── WorkgroupConfig ──────────────────────────────────────────────────

    #[test]
    fn workgroup_config_new_1d() {
        let wg = WorkgroupConfig::new_1d(256);
        assert_eq!(wg.local_x, 256);
        assert_eq!(wg.local_y, 1);
        assert_eq!(wg.local_z, 1);
        assert_eq!(wg.total_size(), 256);
    }

    #[test]
    fn workgroup_config_new_2d() {
        let wg = WorkgroupConfig::new_2d(16, 16);
        assert_eq!(wg.total_size(), 256);
    }

    #[test]
    fn workgroup_config_new_3d() {
        let wg = WorkgroupConfig::new(8, 8, 4);
        assert_eq!(wg.total_size(), 256);
        assert_eq!(wg.as_array(), [8, 8, 4]);
    }

    #[test]
    #[should_panic(expected = "workgroup dimensions must be > 0")]
    fn workgroup_config_zero_panics() {
        WorkgroupConfig::new(0, 1, 1);
    }

    #[test]
    fn workgroup_config_subgroups() {
        let wg = WorkgroupConfig::new_1d(256);
        assert_eq!(wg.subgroups(16), 16);
        assert_eq!(wg.subgroups(32), 8);
        assert_eq!(wg.subgroups(8), 32);
    }

    #[test]
    fn workgroup_config_subgroup_aligned() {
        let wg = WorkgroupConfig::new_1d(256);
        assert!(wg.is_subgroup_aligned(16));
        assert!(wg.is_subgroup_aligned(32));
        assert!(!wg.is_subgroup_aligned(13));
    }

    #[test]
    fn workgroup_config_non_aligned() {
        let wg = WorkgroupConfig::new_1d(100);
        assert!(!wg.is_subgroup_aligned(16));
        assert_eq!(wg.subgroups(16), 7); // ceil(100/16) = 7
    }

    #[test]
    fn workgroup_config_display_1d() {
        let wg = WorkgroupConfig::new_1d(128);
        assert_eq!(format!("{wg}"), "(128,)");
    }

    #[test]
    fn workgroup_config_display_2d() {
        let wg = WorkgroupConfig::new_2d(16, 8);
        assert_eq!(format!("{wg}"), "(16, 8)");
    }

    #[test]
    fn workgroup_config_display_3d() {
        let wg = WorkgroupConfig::new(4, 4, 4);
        assert_eq!(format!("{wg}"), "(4, 4, 4)");
    }

    #[test]
    fn workgroup_config_eq_and_hash() {
        use std::collections::HashSet;
        let a = WorkgroupConfig::new_1d(64);
        let b = WorkgroupConfig::new_1d(64);
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    // ── A770WorkgroupProfile ─────────────────────────────────────────────

    #[test]
    fn a770_defaults() {
        let p = A770WorkgroupProfile::default();
        assert_eq!(p.max_workgroup_size, 1024);
        assert_eq!(p.xe_cores, 32);
        assert_eq!(p.threads_per_core, 8);
        assert_eq!(p.slm_bytes_per_core, 65536);
        assert_eq!(p.preferred_subgroup_size, 16);
        assert_eq!(p.subgroup_sizes, vec![8, 16, 32]);
    }

    #[test]
    fn a770_max_wavefronts() {
        let p = A770WorkgroupProfile::default();
        assert_eq!(p.max_wavefronts_per_core(), 8);
        assert_eq!(p.max_wavefronts_total(), 256);
    }

    #[test]
    fn a770_subgroup_support() {
        let p = A770WorkgroupProfile::default();
        assert!(p.is_subgroup_size_supported(8));
        assert!(p.is_subgroup_size_supported(16));
        assert!(p.is_subgroup_size_supported(32));
        assert!(!p.is_subgroup_size_supported(64));
        assert!(!p.is_subgroup_size_supported(4));
    }

    #[test]
    fn a770_total_slm() {
        let p = A770WorkgroupProfile::default();
        assert_eq!(p.total_slm_bytes(), 32 * 65536);
    }

    // ── RegisterPressure ─────────────────────────────────────────────────

    #[test]
    fn register_pressure_low() {
        let p = A770WorkgroupProfile::default();
        let rp = RegisterPressure::new(16, &p);
        assert!(rp.can_launch());
        assert_eq!(rp.registers_used(), 16);
        assert!((rp.register_utilization() - 0.125).abs() < f64::EPSILON);
    }

    #[test]
    fn register_pressure_max() {
        let p = A770WorkgroupProfile::default();
        // 200 regs > 128 → can't launch
        let rp = RegisterPressure::new(200, &p);
        assert!(!rp.can_launch());
        assert_eq!(rp.max_subgroups_per_thread(), 0);
    }

    #[test]
    fn register_pressure_exact_fit() {
        let p = A770WorkgroupProfile::default();
        // 128 regs = 128 → exact fit
        let rp = RegisterPressure::new(128, &p);
        assert!(rp.can_launch());
        assert!((rp.register_utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn register_pressure_zero_regs() {
        let p = A770WorkgroupProfile::default();
        let rp = RegisterPressure::new(0, &p);
        assert!(rp.can_launch());
        assert!((rp.register_utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn register_pressure_small_subgroup() {
        let rp =
            RegisterPressure { registers_per_item: 4, registers_per_thread: 128, subgroup_size: 8 };
        assert_eq!(rp.registers_used(), 4);
        assert!(rp.can_launch());
        assert!((rp.register_utilization() - (4.0 / 128.0)).abs() < f64::EPSILON);
    }

    // ── SharedMemoryPlanner ──────────────────────────────────────────────

    #[test]
    fn slm_planner_no_usage() {
        let p = A770WorkgroupProfile::default();
        let planner = SharedMemoryPlanner::new(&p);
        assert_eq!(planner.concurrent_workgroups(0), 8);
    }

    #[test]
    fn slm_planner_half_usage() {
        let p = A770WorkgroupProfile::default();
        let planner = SharedMemoryPlanner::new(&p);
        // 32768 bytes = half of 64 KiB → 2 workgroups
        assert_eq!(planner.concurrent_workgroups(32768), 2);
    }

    #[test]
    fn slm_planner_full_usage() {
        let p = A770WorkgroupProfile::default();
        let planner = SharedMemoryPlanner::new(&p);
        assert_eq!(planner.concurrent_workgroups(65536), 1);
    }

    #[test]
    fn slm_planner_exceeds() {
        let p = A770WorkgroupProfile::default();
        let planner = SharedMemoryPlanner::new(&p);
        assert!(!planner.fits(65537));
        assert!(planner.fits(65536));
    }

    #[test]
    fn slm_budget_for_occupancy() {
        let p = A770WorkgroupProfile::default();
        let planner = SharedMemoryPlanner::new(&p);
        assert_eq!(planner.budget_for_occupancy(4), Some(16384));
        assert_eq!(planner.budget_for_occupancy(1), Some(65536));
        assert_eq!(planner.budget_for_occupancy(0), None);
        assert_eq!(planner.budget_for_occupancy(9), None);
    }

    #[test]
    fn slm_max_per_workgroup() {
        let p = A770WorkgroupProfile::default();
        let planner = SharedMemoryPlanner::new(&p);
        assert_eq!(planner.max_slm_per_workgroup(), 65536);
    }

    // ── OccupancyCalculator ──────────────────────────────────────────────

    #[test]
    fn occupancy_full_core_single_wg() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(128);
        // 128 items / 16 sg = 8 subgroups → fills all 8 threads → 100%
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert!((stats.occupancy - 1.0).abs() < f64::EPSILON);
        assert_eq!(stats.wasted_items, 0);
    }

    #[test]
    fn occupancy_half_core() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(64);
        // 64 / 16 = 4 subgroups per WG.  8 threads / 4 = 2 WGs per core →
        // 2×4 = 8 active wavefronts → 100% occupancy.
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert!((stats.occupancy - 1.0).abs() < 0.01);
        assert_eq!(stats.subgroups_per_workgroup, 4);
        assert_eq!(stats.workgroups_per_core, 2);
    }

    #[test]
    fn occupancy_single_subgroup() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(16);
        // 16 / 16 = 1 subgroup per WG.  8 threads / 1 = 8 WGs per core →
        // 8×1 = 8 active wavefronts → 100% occupancy.
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert!((stats.occupancy - 1.0).abs() < 0.01);
        assert_eq!(stats.subgroups_per_workgroup, 1);
        assert_eq!(stats.workgroups_per_core, 8);
    }

    #[test]
    fn occupancy_max_workgroup() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(1024);
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert!(stats.occupancy > 0.0);
        assert_eq!(stats.subgroups_per_workgroup, 64);
    }

    #[test]
    fn occupancy_exceeds_max_returns_none() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(2048);
        let constraints = KernelConstraints::default();
        assert!(calc.calculate(&wg, &constraints).is_none());
    }

    #[test]
    fn occupancy_below_min_returns_none() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(16);
        let constraints = KernelConstraints { min_workgroup_size: 64, ..Default::default() };
        assert!(calc.calculate(&wg, &constraints).is_none());
    }

    #[test]
    fn occupancy_unsupported_subgroup_returns_none() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(64);
        let constraints =
            KernelConstraints { required_subgroup_size: Some(64), ..Default::default() };
        assert!(calc.calculate(&wg, &constraints).is_none());
    }

    #[test]
    fn occupancy_slm_limits_concurrent_wgs() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(16);
        // Request almost all SLM → only 1 workgroup per core
        let constraints =
            KernelConstraints { required_local_memory_bytes: 60000, ..Default::default() };
        let stats = calc.calculate(&wg, &constraints).unwrap();
        assert_eq!(stats.workgroups_per_core, 1);
    }

    #[test]
    fn occupancy_slm_overflow_returns_none() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(16);
        let constraints =
            KernelConstraints { required_local_memory_bytes: 70000, ..Default::default() };
        assert!(calc.calculate(&wg, &constraints).is_none());
    }

    #[test]
    fn occupancy_register_overflow_returns_none() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(16);
        let constraints =
            KernelConstraints { estimated_registers_per_item: 200, ..Default::default() };
        assert!(calc.calculate(&wg, &constraints).is_none());
    }

    #[test]
    fn occupancy_2d_workgroup() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_2d(16, 4);
        // 64 items / 16 = 4 subgroups
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert_eq!(stats.subgroups_per_workgroup, 4);
        assert!(stats.occupancy > 0.0);
    }

    #[test]
    fn occupancy_3d_workgroup() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new(8, 4, 2);
        // 64 items
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert_eq!(stats.workgroup_size, 64);
    }

    #[test]
    fn occupancy_custom_max_workgroup() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(512);
        let constraints = KernelConstraints { max_workgroup_size: 256, ..Default::default() };
        // 512 > 256 → infeasible
        assert!(calc.calculate(&wg, &constraints).is_none());
    }

    #[test]
    fn occupancy_uses_required_subgroup() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(32);
        let constraints =
            KernelConstraints { required_subgroup_size: Some(32), ..Default::default() };
        let stats = calc.calculate(&wg, &constraints).unwrap();
        assert_eq!(stats.subgroup_size, 32);
        assert_eq!(stats.subgroups_per_workgroup, 1);
    }

    #[test]
    fn occupancy_stats_display() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(128);
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        let display = format!("{stats}");
        assert!(display.contains("WG=128"));
        assert!(display.contains("SG=16"));
    }

    #[test]
    fn occupancy_wasted_items_non_aligned() {
        let calc = OccupancyCalculator::a770();
        // 17 items: ceil(17/16) = 2 subgroups, wasted = 32 - 17 = 15
        let wg = WorkgroupConfig::new_1d(17);
        let constraints =
            KernelConstraints { required_subgroup_size: Some(16), ..Default::default() };
        let stats = calc.calculate(&wg, &constraints).unwrap();
        assert_eq!(stats.wasted_items, 15);
        assert!(stats.efficiency < 1.0);
    }

    #[test]
    fn occupancy_multiple_wgs_per_core() {
        let calc = OccupancyCalculator::a770();
        // Small workgroup → many can coexist on one core
        let wg = WorkgroupConfig::new_1d(16);
        let stats = calc.calculate(&wg, &KernelConstraints::default()).unwrap();
        assert!(stats.workgroups_per_core >= 1);
    }

    // ── GlobalSizeAligner ────────────────────────────────────────────────

    #[test]
    fn align_1d_exact() {
        let wg = WorkgroupConfig::new_1d(256);
        assert_eq!(GlobalSizeAligner::align_1d(1024, &wg), 1024);
    }

    #[test]
    fn align_1d_round_up() {
        let wg = WorkgroupConfig::new_1d(256);
        assert_eq!(GlobalSizeAligner::align_1d(1000, &wg), 1024);
    }

    #[test]
    fn align_1d_small_problem() {
        let wg = WorkgroupConfig::new_1d(256);
        assert_eq!(GlobalSizeAligner::align_1d(1, &wg), 256);
    }

    #[test]
    fn align_1d_zero_problem() {
        let wg = WorkgroupConfig::new_1d(16);
        // 0 → clamped to 1 → rounded to 16
        assert_eq!(GlobalSizeAligner::align_1d(0, &wg), 16);
    }

    #[test]
    fn align_2d_exact() {
        let wg = WorkgroupConfig::new_2d(16, 8);
        assert_eq!(GlobalSizeAligner::align_2d(64, 128, &wg), (64, 128));
    }

    #[test]
    fn align_2d_round_up() {
        let wg = WorkgroupConfig::new_2d(16, 8);
        assert_eq!(GlobalSizeAligner::align_2d(60, 100, &wg), (64, 112));
    }

    #[test]
    fn align_3d_round_up() {
        let wg = WorkgroupConfig::new(4, 4, 4);
        assert_eq!(GlobalSizeAligner::align_3d(5, 5, 5, &wg), (8, 8, 8));
    }

    #[test]
    fn align_workgroup_count_1d() {
        let wg = WorkgroupConfig::new_1d(128);
        let global = GlobalSizeAligner::align_1d(1000, &wg);
        assert_eq!(GlobalSizeAligner::workgroup_count_1d(global, &wg), 8);
    }

    #[test]
    fn align_workgroup_count_2d() {
        let wg = WorkgroupConfig::new_2d(16, 8);
        let (gr, gc) = GlobalSizeAligner::align_2d(64, 128, &wg);
        assert_eq!(GlobalSizeAligner::workgroup_count_2d(gr, gc, &wg), 64);
    }

    #[test]
    fn align_wasted_1d() {
        let wg = WorkgroupConfig::new_1d(256);
        assert_eq!(GlobalSizeAligner::wasted_1d(1000, &wg), 24);
        assert_eq!(GlobalSizeAligner::wasted_1d(1024, &wg), 0);
    }

    #[test]
    fn align_efficiency_1d_perfect() {
        let wg = WorkgroupConfig::new_1d(256);
        assert!((GlobalSizeAligner::efficiency_1d(1024, &wg) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn align_efficiency_1d_imperfect() {
        let wg = WorkgroupConfig::new_1d(256);
        let eff = GlobalSizeAligner::efficiency_1d(1000, &wg);
        assert!(eff > 0.9 && eff < 1.0);
    }

    #[test]
    fn align_efficiency_2d() {
        let wg = WorkgroupConfig::new_2d(16, 8);
        let eff = GlobalSizeAligner::efficiency_2d(64, 128, &wg);
        assert!((eff - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn align_no_leftover_1d() {
        let wg = WorkgroupConfig::new_1d(64);
        for problem in [1, 7, 63, 64, 65, 128, 1000, 4096, 100000] {
            let aligned = GlobalSizeAligner::align_1d(problem, &wg);
            assert_eq!(aligned % wg.local_x, 0, "problem={problem}");
            assert!(aligned >= problem.max(1), "problem={problem}");
        }
    }

    #[test]
    fn align_no_leftover_2d() {
        let wg = WorkgroupConfig::new_2d(16, 4);
        for (r, c) in [(1, 1), (3, 7), (16, 16), (100, 100)] {
            let (ar, ac) = GlobalSizeAligner::align_2d(r, c, &wg);
            assert_eq!(ar % wg.local_y, 0, "rows={r}");
            assert_eq!(ac % wg.local_x, 0, "cols={c}");
        }
    }

    // ── WorkgroupOptimizer ───────────────────────────────────────────────

    #[test]
    fn optimizer_1d_basic() {
        let opt = WorkgroupOptimizer::a770();
        let (wg, stats) = opt.optimize_1d(4096, &KernelConstraints::default()).unwrap();
        assert!(wg.total_size() <= 1024);
        assert!(wg.total_size() >= 16);
        assert!(stats.occupancy > 0.0);
    }

    #[test]
    fn optimizer_1d_small_problem() {
        let opt = WorkgroupOptimizer::a770();
        let (wg, _) = opt.optimize_1d(8, &KernelConstraints::default()).unwrap();
        assert!(wg.total_size() <= 1024);
        assert!(wg.is_subgroup_aligned(16));
    }

    #[test]
    fn optimizer_1d_very_large_problem() {
        let opt = WorkgroupOptimizer::a770();
        let (wg, stats) = opt.optimize_1d(10_000_000, &KernelConstraints::default()).unwrap();
        assert!(wg.total_size() <= 1024);
        assert!(stats.efficiency > 0.0);
    }

    #[test]
    fn optimizer_1d_respects_max_constraint() {
        let opt = WorkgroupOptimizer::a770();
        let constraints = KernelConstraints { max_workgroup_size: 128, ..Default::default() };
        let (wg, _) = opt.optimize_1d(4096, &constraints).unwrap();
        assert!(wg.total_size() <= 128);
    }

    #[test]
    fn optimizer_1d_respects_min_constraint() {
        let opt = WorkgroupOptimizer::a770();
        let constraints = KernelConstraints { min_workgroup_size: 64, ..Default::default() };
        let (wg, _) = opt.optimize_1d(4096, &constraints).unwrap();
        assert!(wg.total_size() >= 64);
    }

    #[test]
    fn optimizer_1d_respects_subgroup_constraint() {
        let opt = WorkgroupOptimizer::a770();
        let constraints =
            KernelConstraints { required_subgroup_size: Some(32), ..Default::default() };
        let (wg, stats) = opt.optimize_1d(4096, &constraints).unwrap();
        assert_eq!(stats.subgroup_size, 32);
        assert!(wg.total_size() % 32 == 0);
    }

    #[test]
    fn optimizer_2d_basic() {
        let opt = WorkgroupOptimizer::a770();
        let (wg, stats) = opt.optimize_2d(512, 512, &KernelConstraints::default()).unwrap();
        assert!(wg.total_size() <= 1024);
        assert!(stats.occupancy > 0.0);
    }

    #[test]
    fn optimizer_2d_nonsquare() {
        let opt = WorkgroupOptimizer::a770();
        let (wg, _) = opt.optimize_2d(4, 8192, &KernelConstraints::default()).unwrap();
        assert!(wg.total_size() <= 1024);
    }

    #[test]
    fn optimizer_enumerate_1d() {
        let opt = WorkgroupOptimizer::a770();
        let results = opt.enumerate_1d(&KernelConstraints::default());
        // With sg=16, max=1024 → 64 feasible sizes
        assert!(!results.is_empty());
        // Sorted by occupancy descending
        for pair in results.windows(2) {
            assert!(pair[0].1.occupancy >= pair[1].1.occupancy);
        }
    }

    #[test]
    fn optimizer_enumerate_constrained() {
        let opt = WorkgroupOptimizer::a770();
        let constraints = KernelConstraints {
            min_workgroup_size: 64,
            max_workgroup_size: 256,
            ..Default::default()
        };
        let results = opt.enumerate_1d(&constraints);
        for (wg, _) in &results {
            assert!(wg.total_size() >= 64);
            assert!(wg.total_size() <= 256);
        }
    }

    // ── Property-style: selected ≤ device max ────────────────────────────

    #[test]
    fn property_selected_within_device_max() {
        let opt = WorkgroupOptimizer::a770();
        let profile = A770WorkgroupProfile::default();
        for size in [1, 7, 16, 100, 256, 1024, 4096, 100_000, 1_000_000] {
            if let Some((wg, _)) = opt.optimize_1d(size, &KernelConstraints::default()) {
                assert!(
                    wg.total_size() <= profile.max_workgroup_size,
                    "size={size} wg={}",
                    wg.total_size()
                );
            }
        }
    }

    #[test]
    fn property_selected_2d_within_device_max() {
        let opt = WorkgroupOptimizer::a770();
        let profile = A770WorkgroupProfile::default();
        for (r, c) in [(1, 1), (4, 4), (16, 16), (100, 100), (512, 512), (2048, 2048)] {
            if let Some((wg, _)) = opt.optimize_2d(r, c, &KernelConstraints::default()) {
                assert!(
                    wg.total_size() <= profile.max_workgroup_size,
                    "r={r} c={c} wg={}",
                    wg.total_size()
                );
            }
        }
    }

    #[test]
    fn property_global_always_multiple_of_local() {
        let opt = WorkgroupOptimizer::a770();
        for size in [1, 13, 64, 255, 1024, 99999] {
            if let Some((wg, _)) = opt.optimize_1d(size, &KernelConstraints::default()) {
                let global = GlobalSizeAligner::align_1d(size, &wg);
                assert_eq!(global % wg.local_x, 0, "size={size}");
            }
        }
    }

    #[test]
    fn property_occupancy_bounded() {
        let opt = WorkgroupOptimizer::a770();
        for size in [16, 32, 64, 128, 256, 512, 1024] {
            if let Some((_, stats)) = opt.optimize_1d(size, &KernelConstraints::default()) {
                assert!(stats.occupancy >= 0.0 && stats.occupancy <= 1.0);
                assert!(stats.efficiency >= 0.0 && stats.efficiency <= 1.0);
            }
        }
    }

    #[test]
    fn property_wasted_items_bounded() {
        let calc = OccupancyCalculator::a770();
        for size in 1..=100 {
            let wg = WorkgroupConfig::new_1d(size);
            if let Some(stats) = calc.calculate(&wg, &KernelConstraints::default()) {
                assert!(stats.wasted_items < 16, "size={size} wasted={}", stats.wasted_items);
            }
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn edge_problem_smaller_than_workgroup() {
        let opt = WorkgroupOptimizer::a770();
        // Problem size 1: still gets a valid workgroup
        let result = opt.optimize_1d(1, &KernelConstraints::default());
        assert!(result.is_some());
        let (wg, _) = result.unwrap();
        assert!(wg.total_size() >= 16); // at least one subgroup
    }

    #[test]
    fn edge_problem_exactly_subgroup() {
        let opt = WorkgroupOptimizer::a770();
        let result = opt.optimize_1d(16, &KernelConstraints::default());
        assert!(result.is_some());
    }

    #[test]
    fn edge_impossible_constraints() {
        let opt = WorkgroupOptimizer::a770();
        // min > max → no feasible config
        let constraints = KernelConstraints {
            min_workgroup_size: 512,
            max_workgroup_size: 128,
            ..Default::default()
        };
        assert!(opt.optimize_1d(1024, &constraints).is_none());
    }

    #[test]
    fn edge_slm_limits_workgroup() {
        let opt = WorkgroupOptimizer::a770();
        let constraints =
            KernelConstraints { required_local_memory_bytes: 32768, ..Default::default() };
        // Still finds a valid config despite SLM pressure
        let result = opt.optimize_1d(4096, &constraints);
        assert!(result.is_some());
    }

    #[test]
    fn edge_register_pressure_limits() {
        let opt = WorkgroupOptimizer::a770();
        // 120 regs fits within 128 GRF
        let constraints =
            KernelConstraints { estimated_registers_per_item: 120, ..Default::default() };
        let result = opt.optimize_1d(4096, &constraints);
        assert!(result.is_some());
    }

    #[test]
    fn edge_register_pressure_too_high() {
        let opt = WorkgroupOptimizer::a770();
        // 200 regs exceeds 128 GRF → no config works
        let constraints =
            KernelConstraints { estimated_registers_per_item: 200, ..Default::default() };
        let result = opt.optimize_1d(4096, &constraints);
        assert!(result.is_none());
    }

    #[test]
    fn edge_barriers_flag_accepted() {
        let calc = OccupancyCalculator::a770();
        let wg = WorkgroupConfig::new_1d(128);
        let constraints = KernelConstraints { uses_barriers: true, ..Default::default() };
        // Barriers flag is informational; compute still succeeds
        assert!(calc.calculate(&wg, &constraints).is_some());
    }

    // ── KernelConstraints defaults ───────────────────────────────────────

    #[test]
    fn kernel_constraints_default() {
        let kc = KernelConstraints::default();
        assert_eq!(kc.min_workgroup_size, 1);
        assert_eq!(kc.max_workgroup_size, 0);
        assert_eq!(kc.required_local_memory_bytes, 0);
        assert!(!kc.uses_barriers);
        assert!(kc.required_subgroup_size.is_none());
        assert_eq!(kc.estimated_registers_per_item, 32);
    }

    // ── round_up helper ──────────────────────────────────────────────────

    #[test]
    fn round_up_exact() {
        assert_eq!(round_up(256, 256), 256);
    }

    #[test]
    fn round_up_non_multiple() {
        assert_eq!(round_up(100, 64), 128);
    }

    #[test]
    fn round_up_one() {
        assert_eq!(round_up(1, 16), 16);
    }

    #[test]
    fn round_up_zero_multiple() {
        assert_eq!(round_up(100, 0), 100);
    }

    #[test]
    fn round_up_zero_value() {
        assert_eq!(round_up(0, 16), 0);
    }
}
