//! Kernel dispatch planning for Intel Arc A770 GPU.
//!
//! Computes optimal NDRange parameters, maps operations to hardware resources
//! (EUs, subslices, SLM), and schedules kernel launches with dependency
//! awareness. All implementations are CPU reference code — no OpenCL runtime
//! required.

use std::fmt;

// ---------------------------------------------------------------------------
// DispatchDimensions — 3-D NDRange parameters
// ---------------------------------------------------------------------------

/// Three-dimensional NDRange dispatch parameters for an OpenCL kernel launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchDimensions {
    /// Global work size per dimension `[x, y, z]`.
    pub global_work_size: [usize; 3],
    /// Local work size per dimension `[x, y, z]`.
    pub local_work_size: [usize; 3],
    /// Global offset per dimension `[x, y, z]`.
    pub offset: [usize; 3],
}

impl DispatchDimensions {
    /// Create a 1-D dispatch (y and z sizes set to 1).
    pub fn new_1d(global_x: usize, local_x: usize) -> Self {
        Self {
            global_work_size: [global_x, 1, 1],
            local_work_size: [local_x, 1, 1],
            offset: [0, 0, 0],
        }
    }

    /// Create a 2-D dispatch (z size set to 1).
    pub fn new_2d(global: [usize; 2], local: [usize; 2]) -> Self {
        Self {
            global_work_size: [global[0], global[1], 1],
            local_work_size: [local[0], local[1], 1],
            offset: [0, 0, 0],
        }
    }

    /// Create a full 3-D dispatch.
    pub fn new_3d(global: [usize; 3], local: [usize; 3]) -> Self {
        Self { global_work_size: global, local_work_size: local, offset: [0, 0, 0] }
    }

    /// Total number of work items across all dimensions.
    pub fn total_work_items(&self) -> usize {
        self.global_work_size.iter().product()
    }

    /// Total number of work groups across all dimensions.
    pub fn total_work_groups(&self) -> usize {
        self.global_work_size
            .iter()
            .zip(self.local_work_size.iter())
            .map(|(g, l)| if *l == 0 { 0 } else { g / l })
            .product()
    }

    /// Work-group size (product of local dimensions).
    pub fn workgroup_size(&self) -> usize {
        self.local_work_size.iter().product()
    }

    /// Number of active dimensions (dimensions with size > 1).
    pub fn active_dimensions(&self) -> usize {
        self.global_work_size.iter().filter(|&&s| s > 1).count().max(1)
    }

    /// Validate that global sizes are evenly divisible by local sizes.
    pub fn is_valid(&self) -> bool {
        for i in 0..3 {
            if self.local_work_size[i] == 0 {
                return false;
            }
            if !self.global_work_size[i].is_multiple_of(self.local_work_size[i]) {
                return false;
            }
        }
        true
    }
}

impl fmt::Display for DispatchDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dims = self.active_dimensions();
        write!(
            f,
            "{}D global={:?} local={:?}",
            dims,
            &self.global_work_size[..dims],
            &self.local_work_size[..dims],
        )
    }
}

// ---------------------------------------------------------------------------
// A770Resources — Intel Arc A770 hardware spec
// ---------------------------------------------------------------------------

/// Hardware resource description for the Intel Arc A770 GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A770Resources {
    /// Number of Execution Units.
    pub eu_count: usize,
    /// Number of subslices (Xe-cores).
    pub subslices: usize,
    /// Shared Local Memory per subslice in bytes.
    pub slm_per_subslice: usize,
    /// Maximum work items per work group.
    pub max_workgroup: usize,
    /// Supported subgroup (SIMD) sizes.
    pub subgroup_sizes: Vec<usize>,
}

impl Default for A770Resources {
    fn default() -> Self {
        Self {
            eu_count: 512,
            subslices: 32,
            slm_per_subslice: 65536,
            max_workgroup: 1024,
            subgroup_sizes: vec![8, 16, 32],
        }
    }
}

impl A770Resources {
    /// Total SLM across the entire GPU.
    pub fn total_slm(&self) -> usize {
        self.subslices * self.slm_per_subslice
    }

    /// EUs per subslice.
    pub fn eus_per_subslice(&self) -> usize {
        if self.subslices == 0 {
            return 0;
        }
        self.eu_count / self.subslices
    }

    /// Whether a given subgroup size is supported.
    pub fn supports_subgroup(&self, size: usize) -> bool {
        self.subgroup_sizes.contains(&size)
    }

    /// Largest supported subgroup size.
    pub fn max_subgroup_size(&self) -> usize {
        self.subgroup_sizes.iter().copied().max().unwrap_or(8)
    }
}

impl fmt::Display for A770Resources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "A770(EUs={}, subslices={}, SLM={}KB/ss, max_wg={})",
            self.eu_count,
            self.subslices,
            self.slm_per_subslice / 1024,
            self.max_workgroup,
        )
    }
}

// ---------------------------------------------------------------------------
// DispatchConstraint
// ---------------------------------------------------------------------------

/// Constraints that a dispatch plan must satisfy.
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchConstraint {
    /// Workgroup size must not exceed this value.
    MaxWorkgroup(usize),
    /// Prefer a specific subgroup (SIMD) size.
    PreferredSubgroup(usize),
    /// Minimum GPU occupancy (0.0–1.0).
    MinOccupancy(f32),
    /// Maximum SLM usage in bytes per workgroup.
    SLMBudget(usize),
}

impl fmt::Display for DispatchConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxWorkgroup(n) => write!(f, "MaxWorkgroup({n})"),
            Self::PreferredSubgroup(n) => write!(f, "PreferredSubgroup({n})"),
            Self::MinOccupancy(o) => write!(f, "MinOccupancy({o:.2})"),
            Self::SLMBudget(b) => write!(f, "SLMBudget({b})"),
        }
    }
}

// ---------------------------------------------------------------------------
// DispatchPlan — computed plan for a kernel launch
// ---------------------------------------------------------------------------

/// Result of dispatch planning: dimensions plus analysis metadata.
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// Computed NDRange dimensions.
    pub dimensions: DispatchDimensions,
    /// Which constraints were checked and satisfied.
    pub constraints_satisfied: Vec<bool>,
    /// Estimated GPU occupancy (0.0–1.0).
    pub estimated_occupancy: f32,
    /// Estimated number of hardware waves to complete the dispatch.
    pub estimated_waves: f32,
}

impl DispatchPlan {
    /// Whether all requested constraints are satisfied.
    pub fn all_constraints_satisfied(&self) -> bool {
        self.constraints_satisfied.iter().all(|&s| s)
    }
}

impl fmt::Display for DispatchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DispatchPlan({}, occ={:.1}%, waves={:.1}, ok={})",
            self.dimensions,
            self.estimated_occupancy * 100.0,
            self.estimated_waves,
            self.all_constraints_satisfied(),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Round `value` up to the next multiple of `multiple`.
#[inline]
fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    let remainder = value % multiple;
    if remainder == 0 { value } else { value + multiple - remainder }
}

/// Largest factor of `n` that does not exceed `limit` and is a multiple of
/// `alignment`.
fn largest_aligned_factor(n: usize, limit: usize, alignment: usize) -> usize {
    if alignment == 0 || n == 0 {
        return 1;
    }
    let mut best = alignment;
    let mut candidate = alignment;
    while candidate <= limit && candidate <= n {
        if n.is_multiple_of(candidate) {
            best = candidate;
        }
        candidate += alignment;
    }
    best
}

// ---------------------------------------------------------------------------
// OccupancyCalculator
// ---------------------------------------------------------------------------

/// Estimates GPU occupancy based on workgroup size, register pressure, and
/// SLM consumption.
#[derive(Debug, Clone)]
pub struct OccupancyCalculator {
    resources: A770Resources,
}

impl OccupancyCalculator {
    pub fn new(resources: A770Resources) -> Self {
        Self { resources }
    }

    /// Maximum concurrent workgroups per subslice given SLM budget.
    pub fn max_workgroups_by_slm(&self, slm_per_workgroup: usize) -> usize {
        if slm_per_workgroup == 0 {
            // No SLM used — limited only by other factors; assume 16
            // as a reasonable upper bound for the A770.
            return 16;
        }
        self.resources.slm_per_subslice / slm_per_workgroup
    }

    /// Maximum concurrent workgroups per subslice given register usage.
    /// `regs_per_thread` is the number of GRF registers each work item needs.
    /// A770 has 128 GRF registers per EU thread, with up to 8 threads/EU.
    pub fn max_workgroups_by_registers(
        &self,
        workgroup_size: usize,
        regs_per_thread: usize,
    ) -> usize {
        if workgroup_size == 0 || regs_per_thread == 0 {
            return 0;
        }
        let eus = self.resources.eus_per_subslice();
        // Each EU has 128 registers, supports 8 hardware threads.
        let threads_per_eu = 128usize.min(128 / regs_per_thread).max(1);
        let total_threads = eus * threads_per_eu;
        total_threads / workgroup_size
    }

    /// Estimate occupancy (0.0–1.0).
    ///
    /// `workgroup_size` — work items per workgroup.
    /// `regs_per_thread` — GRF registers per work item (0 = assume light).
    /// `slm_per_workgroup` — SLM bytes per workgroup (0 = none).
    pub fn estimate_occupancy(
        &self,
        workgroup_size: usize,
        regs_per_thread: usize,
        slm_per_workgroup: usize,
    ) -> f32 {
        if workgroup_size == 0 {
            return 0.0;
        }
        let eus = self.resources.eus_per_subslice();
        if eus == 0 {
            return 0.0;
        }

        let max_by_slm = self.max_workgroups_by_slm(slm_per_workgroup);
        let regs = if regs_per_thread == 0 { 32 } else { regs_per_thread };
        let max_by_regs = self.max_workgroups_by_registers(workgroup_size, regs);

        let active_wgs = max_by_slm.min(max_by_regs).max(1);
        let active_threads = active_wgs * workgroup_size;

        // A770: 8 threads per EU is the theoretical maximum occupancy.
        let max_threads_per_ss = eus * 8;
        let occ = active_threads as f32 / max_threads_per_ss as f32;
        occ.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// WaveAnalyzer
// ---------------------------------------------------------------------------

/// Analyzes how many hardware waves are needed to complete a dispatch.
#[derive(Debug, Clone)]
pub struct WaveAnalyzer {
    resources: A770Resources,
}

impl WaveAnalyzer {
    pub fn new(resources: A770Resources) -> Self {
        Self { resources }
    }

    /// Compute the number of waves for a given dispatch.
    ///
    /// A "wave" is one round of concurrent workgroup execution across all
    /// subslices. If there are more workgroups than can run simultaneously,
    /// multiple waves are required.
    pub fn compute_waves(&self, total_workgroups: usize, workgroups_per_subslice: usize) -> f32 {
        if total_workgroups == 0 {
            return 0.0;
        }
        let concurrent = workgroups_per_subslice.max(1);
        let total_concurrent = concurrent * self.resources.subslices;
        total_workgroups as f32 / total_concurrent as f32
    }

    /// Compute waves from a dispatch plan's dimensions with default concurrency.
    pub fn waves_for_dispatch(&self, dims: &DispatchDimensions) -> f32 {
        let wgs = dims.total_work_groups();
        // Assume 1 workgroup per subslice for conservative estimate.
        self.compute_waves(wgs, 1)
    }

    /// Compute waves with SLM-limited concurrency.
    pub fn waves_with_slm(&self, dims: &DispatchDimensions, slm_per_workgroup: usize) -> f32 {
        let wgs = dims.total_work_groups();
        let wgs_per_ss = if slm_per_workgroup == 0 {
            4 // default concurrency when no SLM
        } else {
            (self.resources.slm_per_subslice / slm_per_workgroup).max(1)
        };
        self.compute_waves(wgs, wgs_per_ss)
    }
}

// ---------------------------------------------------------------------------
// DispatchPlanner — constraint-aware optimal dispatch computation
// ---------------------------------------------------------------------------

/// Computes optimal dispatch dimensions for a given operation and A770
/// resources, respecting user-supplied constraints.
#[derive(Debug, Clone)]
pub struct DispatchPlanner {
    resources: A770Resources,
    constraints: Vec<DispatchConstraint>,
}

impl DispatchPlanner {
    /// Create a planner with default A770 resources.
    pub fn new() -> Self {
        Self { resources: A770Resources::default(), constraints: Vec::new() }
    }

    /// Create a planner with custom resources.
    pub fn with_resources(resources: A770Resources) -> Self {
        Self { resources, constraints: Vec::new() }
    }

    /// Add a constraint.
    pub fn add_constraint(&mut self, c: DispatchConstraint) {
        self.constraints.push(c);
    }

    /// Builder-style constraint addition.
    pub fn constraint(mut self, c: DispatchConstraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Access the underlying resources.
    pub fn resources(&self) -> &A770Resources {
        &self.resources
    }

    // ---- effective limits -------------------------------------------------

    /// Effective maximum workgroup size after applying constraints.
    fn effective_max_workgroup(&self) -> usize {
        let hw_max = self.resources.max_workgroup;
        self.constraints.iter().fold(hw_max, |acc, c| match c {
            DispatchConstraint::MaxWorkgroup(n) => acc.min(*n),
            _ => acc,
        })
    }

    /// Preferred subgroup size from constraints, or the largest supported.
    fn preferred_subgroup(&self) -> usize {
        for c in &self.constraints {
            if let DispatchConstraint::PreferredSubgroup(s) = c
                && self.resources.supports_subgroup(*s)
            {
                return *s;
            }
        }
        self.resources.max_subgroup_size()
    }

    /// SLM budget from constraints, or the full per-subslice budget.
    fn slm_budget(&self) -> usize {
        for c in &self.constraints {
            if let DispatchConstraint::SLMBudget(b) = c {
                return *b;
            }
        }
        self.resources.slm_per_subslice
    }

    // ---- planning ---------------------------------------------------------

    /// Plan a 1-D dispatch for `elements` work items.
    pub fn plan_1d(&self, elements: usize) -> DispatchPlan {
        let elements = elements.max(1);
        let sg = self.preferred_subgroup();
        let max_wg = self.effective_max_workgroup();

        // Choose local size: largest multiple of subgroup that divides
        // evenly into a rounded-up global size, up to max_wg.
        let local_x = self.pick_local_size(elements, sg, max_wg);
        let global_x = round_up(elements, local_x);

        let dims = DispatchDimensions::new_1d(global_x, local_x);
        self.build_plan(dims)
    }

    /// Plan a 2-D dispatch for `[rows, cols]`.
    pub fn plan_2d(&self, rows: usize, cols: usize) -> DispatchPlan {
        let rows = rows.max(1);
        let cols = cols.max(1);
        let sg = self.preferred_subgroup();
        let max_wg = self.effective_max_workgroup();

        let local_x = self.pick_local_size(cols, sg, max_wg);
        let budget_y = (max_wg / local_x).max(1);
        let local_y = largest_aligned_factor(round_up(rows, 1), budget_y, 1).max(1);

        let global_x = round_up(cols, local_x);
        let global_y = round_up(rows, local_y);

        let dims = DispatchDimensions::new_2d([global_x, global_y], [local_x, local_y]);
        self.build_plan(dims)
    }

    /// Plan a 3-D dispatch for `[batch, rows, cols]`.
    pub fn plan_3d(&self, batch: usize, rows: usize, cols: usize) -> DispatchPlan {
        let batch = batch.max(1);
        let rows = rows.max(1);
        let cols = cols.max(1);
        let sg = self.preferred_subgroup();
        let max_wg = self.effective_max_workgroup();

        let local_x = self.pick_local_size(cols, sg, max_wg);
        let yz_budget = (max_wg / local_x).max(1);
        let local_y = largest_aligned_factor(round_up(rows, 1), yz_budget, 1).max(1);
        let local_z = (yz_budget / local_y).clamp(1, batch);

        let global_x = round_up(cols, local_x);
        let global_y = round_up(rows, local_y);
        let global_z = round_up(batch, local_z);

        let dims =
            DispatchDimensions::new_3d([global_x, global_y, global_z], [local_x, local_y, local_z]);
        self.build_plan(dims)
    }

    // ---- internals --------------------------------------------------------

    /// Choose a local (workgroup) size for one dimension.
    fn pick_local_size(&self, problem_size: usize, subgroup: usize, max_wg: usize) -> usize {
        // Start from the largest multiple of subgroup ≤ max_wg.
        let cap = (max_wg / subgroup) * subgroup;
        if cap == 0 {
            return 1;
        }
        // Prefer a size that divides the rounded-up global evenly.
        let rounded = round_up(problem_size, subgroup);
        largest_aligned_factor(rounded, cap, subgroup).max(subgroup)
    }

    /// Build the final plan by evaluating all constraints.
    fn build_plan(&self, dims: DispatchDimensions) -> DispatchPlan {
        let calc = OccupancyCalculator::new(self.resources.clone());
        let analyzer = WaveAnalyzer::new(self.resources.clone());

        let wg_size = dims.workgroup_size();
        let slm_budget = self.slm_budget();
        let occupancy = calc.estimate_occupancy(wg_size, 0, slm_budget);
        let waves = analyzer.waves_with_slm(&dims, slm_budget);

        let satisfied: Vec<bool> = self
            .constraints
            .iter()
            .map(|c| match c {
                DispatchConstraint::MaxWorkgroup(max) => wg_size <= *max,
                DispatchConstraint::PreferredSubgroup(sg) => {
                    // Satisfied if local_x is a multiple of sg.
                    dims.local_work_size[0].is_multiple_of(*sg)
                }
                DispatchConstraint::MinOccupancy(min) => occupancy >= *min,
                DispatchConstraint::SLMBudget(budget) => *budget <= self.resources.slm_per_subslice,
            })
            .collect();

        DispatchPlan {
            dimensions: dims,
            constraints_satisfied: satisfied,
            estimated_occupancy: occupancy,
            estimated_waves: waves,
        }
    }
}

impl Default for DispatchPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DispatchPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DispatchPlanner({}, {} constraints)", self.resources, self.constraints.len(),)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_resources() -> A770Resources {
        A770Resources::default()
    }

    fn default_planner() -> DispatchPlanner {
        DispatchPlanner::new()
    }

    /// Assert invariants that every valid dispatch plan must satisfy.
    fn assert_plan_invariants(plan: &DispatchPlan, res: &A770Resources) {
        let dims = &plan.dimensions;
        assert!(dims.is_valid(), "dimensions not valid: {dims}");
        assert!(
            dims.workgroup_size() <= res.max_workgroup,
            "workgroup {} > max {}",
            dims.workgroup_size(),
            res.max_workgroup,
        );
        assert!(
            plan.estimated_occupancy >= 0.0 && plan.estimated_occupancy <= 1.0,
            "occupancy {} out of range",
            plan.estimated_occupancy,
        );
        assert!(plan.estimated_waves >= 0.0, "negative waves: {}", plan.estimated_waves,);
    }

    // -----------------------------------------------------------------------
    // DispatchDimensions
    // -----------------------------------------------------------------------

    #[test]
    fn dimensions_1d_basic() {
        let d = DispatchDimensions::new_1d(256, 32);
        assert_eq!(d.global_work_size, [256, 1, 1]);
        assert_eq!(d.local_work_size, [32, 1, 1]);
        assert_eq!(d.offset, [0, 0, 0]);
        assert_eq!(d.total_work_items(), 256);
        assert_eq!(d.total_work_groups(), 8);
        assert_eq!(d.workgroup_size(), 32);
        assert_eq!(d.active_dimensions(), 1);
        assert!(d.is_valid());
    }

    #[test]
    fn dimensions_2d_basic() {
        let d = DispatchDimensions::new_2d([128, 64], [16, 8]);
        assert_eq!(d.global_work_size, [128, 64, 1]);
        assert_eq!(d.local_work_size, [16, 8, 1]);
        assert_eq!(d.total_work_items(), 128 * 64);
        assert_eq!(d.total_work_groups(), 8 * 8);
        assert_eq!(d.workgroup_size(), 128);
        assert_eq!(d.active_dimensions(), 2);
        assert!(d.is_valid());
    }

    #[test]
    fn dimensions_3d_basic() {
        let d = DispatchDimensions::new_3d([64, 32, 8], [8, 4, 2]);
        assert_eq!(d.total_work_items(), 64 * 32 * 8);
        assert_eq!(d.total_work_groups(), 8 * 8 * 4);
        assert_eq!(d.workgroup_size(), 64);
        assert_eq!(d.active_dimensions(), 3);
        assert!(d.is_valid());
    }

    #[test]
    fn dimensions_invalid_not_divisible() {
        let d = DispatchDimensions::new_1d(100, 32);
        assert!(!d.is_valid());
    }

    #[test]
    fn dimensions_zero_local_invalid() {
        let d = DispatchDimensions {
            global_work_size: [256, 1, 1],
            local_work_size: [0, 1, 1],
            offset: [0, 0, 0],
        };
        assert!(!d.is_valid());
    }

    #[test]
    fn dimensions_display() {
        let d = DispatchDimensions::new_1d(256, 32);
        let s = format!("{d}");
        assert!(s.contains("1D"));
        assert!(s.contains("256"));
    }

    // -----------------------------------------------------------------------
    // A770Resources
    // -----------------------------------------------------------------------

    #[test]
    fn a770_defaults() {
        let r = default_resources();
        assert_eq!(r.eu_count, 512);
        assert_eq!(r.subslices, 32);
        assert_eq!(r.slm_per_subslice, 65536);
        assert_eq!(r.max_workgroup, 1024);
        assert_eq!(r.subgroup_sizes, vec![8, 16, 32]);
    }

    #[test]
    fn a770_total_slm() {
        let r = default_resources();
        assert_eq!(r.total_slm(), 32 * 65536);
    }

    #[test]
    fn a770_eus_per_subslice() {
        let r = default_resources();
        assert_eq!(r.eus_per_subslice(), 16);
    }

    #[test]
    fn a770_supports_subgroup() {
        let r = default_resources();
        assert!(r.supports_subgroup(8));
        assert!(r.supports_subgroup(16));
        assert!(r.supports_subgroup(32));
        assert!(!r.supports_subgroup(64));
        assert!(!r.supports_subgroup(4));
    }

    #[test]
    fn a770_max_subgroup_size() {
        let r = default_resources();
        assert_eq!(r.max_subgroup_size(), 32);
    }

    #[test]
    fn a770_display() {
        let r = default_resources();
        let s = format!("{r}");
        assert!(s.contains("512"));
        assert!(s.contains("A770"));
    }

    // -----------------------------------------------------------------------
    // DispatchConstraint
    // -----------------------------------------------------------------------

    #[test]
    fn constraint_display() {
        assert!(format!("{}", DispatchConstraint::MaxWorkgroup(256)).contains("256"));
        assert!(format!("{}", DispatchConstraint::PreferredSubgroup(16)).contains("16"));
        assert!(format!("{}", DispatchConstraint::MinOccupancy(0.5)).contains("0.50"));
        assert!(format!("{}", DispatchConstraint::SLMBudget(4096)).contains("4096"));
    }

    // -----------------------------------------------------------------------
    // OccupancyCalculator
    // -----------------------------------------------------------------------

    #[test]
    fn occupancy_no_slm_no_reg_pressure() {
        let calc = OccupancyCalculator::new(default_resources());
        let occ = calc.estimate_occupancy(256, 0, 0);
        assert!(occ > 0.0, "occ={occ}");
        assert!(occ <= 1.0, "occ={occ}");
    }

    #[test]
    fn occupancy_heavy_slm() {
        let calc = OccupancyCalculator::new(default_resources());
        // Use half the SLM per workgroup → at most 2 concurrent WGs.
        let occ = calc.estimate_occupancy(256, 0, 32768);
        assert!(occ > 0.0);
        assert!(occ <= 1.0);
    }

    #[test]
    fn occupancy_heavy_registers() {
        let calc = OccupancyCalculator::new(default_resources());
        // 128 registers per thread = 1 thread/EU.
        let occ = calc.estimate_occupancy(16, 128, 0);
        assert!(occ > 0.0);
        assert!(occ <= 1.0);
    }

    #[test]
    fn occupancy_zero_workgroup_returns_zero() {
        let calc = OccupancyCalculator::new(default_resources());
        assert_eq!(calc.estimate_occupancy(0, 0, 0), 0.0);
    }

    #[test]
    fn occupancy_max_workgroups_by_slm_zero() {
        let calc = OccupancyCalculator::new(default_resources());
        assert_eq!(calc.max_workgroups_by_slm(0), 16);
    }

    #[test]
    fn occupancy_max_workgroups_by_slm_half() {
        let calc = OccupancyCalculator::new(default_resources());
        assert_eq!(calc.max_workgroups_by_slm(32768), 2);
    }

    #[test]
    fn occupancy_max_workgroups_by_slm_full() {
        let calc = OccupancyCalculator::new(default_resources());
        assert_eq!(calc.max_workgroups_by_slm(65536), 1);
    }

    #[test]
    fn occupancy_larger_workgroup_different_occupancy() {
        let calc = OccupancyCalculator::new(default_resources());
        let small_wg = calc.estimate_occupancy(32, 0, 0);
        let large_wg = calc.estimate_occupancy(1024, 0, 0);
        // Both must produce valid occupancy values.
        assert!(small_wg > 0.0 && small_wg <= 1.0);
        assert!(large_wg > 0.0 && large_wg <= 1.0);
    }

    // -----------------------------------------------------------------------
    // WaveAnalyzer
    // -----------------------------------------------------------------------

    #[test]
    fn waves_zero_workgroups() {
        let analyzer = WaveAnalyzer::new(default_resources());
        assert_eq!(analyzer.compute_waves(0, 1), 0.0);
    }

    #[test]
    fn waves_single_wave() {
        let analyzer = WaveAnalyzer::new(default_resources());
        // 32 subslices * 1 wg/ss = 32 concurrent workgroups.
        let waves = analyzer.compute_waves(32, 1);
        assert!((waves - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn waves_multiple_waves() {
        let analyzer = WaveAnalyzer::new(default_resources());
        // 64 workgroups with 1/ss → 2 waves.
        let waves = analyzer.compute_waves(64, 1);
        assert!((waves - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn waves_fractional() {
        let analyzer = WaveAnalyzer::new(default_resources());
        // 48 workgroups with 1/ss → 48/32 = 1.5 waves.
        let waves = analyzer.compute_waves(48, 1);
        assert!((waves - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn waves_with_slm_limited_concurrency() {
        let analyzer = WaveAnalyzer::new(default_resources());
        let dims = DispatchDimensions::new_1d(1024, 32);
        // 1024/32 = 32 workgroups, SLM per WG = 32KB → 2 WGs/ss.
        let waves = analyzer.waves_with_slm(&dims, 32768);
        // 32 wgs / (2*32) = 0.5 waves.
        assert!((waves - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn waves_for_dispatch_conservative() {
        let analyzer = WaveAnalyzer::new(default_resources());
        let dims = DispatchDimensions::new_1d(1024, 32);
        // 32 workgroups, 1/ss → 1.0 wave.
        let waves = analyzer.waves_for_dispatch(&dims);
        assert!((waves - 1.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // DispatchPlanner — 1D
    // -----------------------------------------------------------------------

    #[test]
    fn plan_1d_small_tensor() {
        let planner = default_planner();
        let plan = planner.plan_1d(64);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 64);
    }

    #[test]
    fn plan_1d_medium_tensor() {
        let planner = default_planner();
        let plan = planner.plan_1d(2048);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 2048);
    }

    #[test]
    fn plan_1d_large_tensor() {
        let planner = default_planner();
        let plan = planner.plan_1d(1_000_000);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 1_000_000);
    }

    #[test]
    fn plan_1d_size_one() {
        let planner = default_planner();
        let plan = planner.plan_1d(1);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 1);
    }

    #[test]
    fn plan_1d_non_power_of_two() {
        let planner = default_planner();
        let plan = planner.plan_1d(1000);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 1000);
    }

    // -----------------------------------------------------------------------
    // DispatchPlanner — 2D
    // -----------------------------------------------------------------------

    #[test]
    fn plan_2d_square() {
        let planner = default_planner();
        let plan = planner.plan_2d(256, 256);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 256);
        assert!(plan.dimensions.global_work_size[1] >= 256);
    }

    #[test]
    fn plan_2d_rectangular() {
        let planner = default_planner();
        let plan = planner.plan_2d(64, 2048);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 2048);
        assert!(plan.dimensions.global_work_size[1] >= 64);
    }

    #[test]
    fn plan_2d_size_one_row() {
        let planner = default_planner();
        let plan = planner.plan_2d(1, 512);
        assert_plan_invariants(&plan, planner.resources());
    }

    #[test]
    fn plan_2d_non_power_of_two() {
        let planner = default_planner();
        let plan = planner.plan_2d(100, 300);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 300);
        assert!(plan.dimensions.global_work_size[1] >= 100);
    }

    // -----------------------------------------------------------------------
    // DispatchPlanner — 3D
    // -----------------------------------------------------------------------

    #[test]
    fn plan_3d_batched() {
        let planner = default_planner();
        let plan = planner.plan_3d(4, 128, 256);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 256);
        assert!(plan.dimensions.global_work_size[1] >= 128);
        assert!(plan.dimensions.global_work_size[2] >= 4);
    }

    #[test]
    fn plan_3d_single_batch() {
        let planner = default_planner();
        let plan = planner.plan_3d(1, 64, 64);
        assert_plan_invariants(&plan, planner.resources());
    }

    #[test]
    fn plan_3d_all_ones() {
        let planner = default_planner();
        let plan = planner.plan_3d(1, 1, 1);
        assert_plan_invariants(&plan, planner.resources());
    }

    // -----------------------------------------------------------------------
    // Constraint satisfaction
    // -----------------------------------------------------------------------

    #[test]
    fn constraint_max_workgroup_satisfied() {
        let planner = default_planner().constraint(DispatchConstraint::MaxWorkgroup(1024));
        let plan = planner.plan_1d(4096);
        assert!(plan.all_constraints_satisfied());
        assert!(plan.dimensions.workgroup_size() <= 1024);
    }

    #[test]
    fn constraint_max_workgroup_tight() {
        let planner = default_planner().constraint(DispatchConstraint::MaxWorkgroup(32));
        let plan = planner.plan_1d(4096);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.workgroup_size() <= 32);
        assert!(plan.all_constraints_satisfied());
    }

    #[test]
    fn constraint_preferred_subgroup_16() {
        let planner = default_planner().constraint(DispatchConstraint::PreferredSubgroup(16));
        let plan = planner.plan_1d(1024);
        assert!(plan.all_constraints_satisfied());
        assert_eq!(plan.dimensions.local_work_size[0] % 16, 0);
    }

    #[test]
    fn constraint_preferred_subgroup_8() {
        let planner = default_planner().constraint(DispatchConstraint::PreferredSubgroup(8));
        let plan = planner.plan_1d(1024);
        assert!(plan.all_constraints_satisfied());
        assert_eq!(plan.dimensions.local_work_size[0] % 8, 0);
    }

    #[test]
    fn constraint_slm_budget_enforced() {
        let planner = default_planner().constraint(DispatchConstraint::SLMBudget(16384));
        let plan = planner.plan_1d(4096);
        assert!(plan.all_constraints_satisfied());
    }

    #[test]
    fn constraint_slm_budget_over_hw_limit() {
        let planner = default_planner().constraint(DispatchConstraint::SLMBudget(100_000));
        let plan = planner.plan_1d(4096);
        // Budget exceeds per-subslice SLM → constraint not satisfied.
        assert!(!plan.all_constraints_satisfied());
    }

    #[test]
    fn multiple_constraints() {
        let planner = default_planner()
            .constraint(DispatchConstraint::MaxWorkgroup(512))
            .constraint(DispatchConstraint::PreferredSubgroup(16))
            .constraint(DispatchConstraint::SLMBudget(8192));
        let plan = planner.plan_1d(8192);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.workgroup_size() <= 512);
        assert!(plan.all_constraints_satisfied());
    }

    // -----------------------------------------------------------------------
    // Subgroup size selection
    // -----------------------------------------------------------------------

    #[test]
    fn subgroup_default_is_max() {
        let planner = default_planner();
        let plan = planner.plan_1d(1024);
        // Default picks max subgroup (32) and local should be multiple.
        assert_eq!(plan.dimensions.local_work_size[0] % 32, 0);
    }

    #[test]
    fn subgroup_override_to_16() {
        let planner = default_planner().constraint(DispatchConstraint::PreferredSubgroup(16));
        let plan = planner.plan_1d(1024);
        assert_eq!(plan.dimensions.local_work_size[0] % 16, 0);
    }

    #[test]
    fn subgroup_override_to_8() {
        let planner = default_planner().constraint(DispatchConstraint::PreferredSubgroup(8));
        let plan = planner.plan_1d(1024);
        assert_eq!(plan.dimensions.local_work_size[0] % 8, 0);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn plan_zero_elements_treated_as_one() {
        let planner = default_planner();
        let plan = planner.plan_1d(0);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 1);
    }

    #[test]
    fn plan_very_large_1d() {
        let planner = default_planner();
        let plan = planner.plan_1d(100_000_000);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 100_000_000);
    }

    #[test]
    fn plan_prime_size() {
        let planner = default_planner();
        let plan = planner.plan_1d(997);
        assert_plan_invariants(&plan, planner.resources());
        assert!(plan.dimensions.global_work_size[0] >= 997);
    }

    // -----------------------------------------------------------------------
    // DispatchPlan display
    // -----------------------------------------------------------------------

    #[test]
    fn plan_display() {
        let planner = default_planner();
        let plan = planner.plan_1d(1024);
        let s = format!("{plan}");
        assert!(s.contains("DispatchPlan"));
        assert!(s.contains("occ="));
        assert!(s.contains("waves="));
    }

    // -----------------------------------------------------------------------
    // Property tests: all plans satisfy A770 constraints
    // -----------------------------------------------------------------------

    #[test]
    fn property_all_1d_plans_valid() {
        let planner = default_planner();
        let res = planner.resources().clone();
        for &size in
            &[1, 2, 7, 16, 31, 32, 33, 64, 100, 255, 256, 512, 1000, 1024, 2048, 4096, 10000, 65536]
        {
            let plan = planner.plan_1d(size);
            assert_plan_invariants(&plan, &res);
            assert!(
                plan.dimensions.global_work_size[0] >= size,
                "global {} < size {size}",
                plan.dimensions.global_work_size[0],
            );
        }
    }

    #[test]
    fn property_all_2d_plans_valid() {
        let planner = default_planner();
        let res = planner.resources().clone();
        for &(r, c) in
            &[(1, 1), (1, 256), (256, 1), (32, 32), (100, 200), (512, 512), (1, 10000), (7, 13)]
        {
            let plan = planner.plan_2d(r, c);
            assert_plan_invariants(&plan, &res);
        }
    }

    #[test]
    fn property_all_3d_plans_valid() {
        let planner = default_planner();
        let res = planner.resources().clone();
        for &(b, r, c) in
            &[(1, 1, 1), (4, 32, 64), (2, 128, 256), (8, 8, 8), (1, 1, 1024), (3, 5, 7)]
        {
            let plan = planner.plan_3d(b, r, c);
            assert_plan_invariants(&plan, &res);
        }
    }

    #[test]
    fn property_workgroup_never_exceeds_max() {
        let planner = default_planner().constraint(DispatchConstraint::MaxWorkgroup(256));
        for &size in &[1, 64, 256, 1024, 4096, 100_000] {
            let plan = planner.plan_1d(size);
            assert!(
                plan.dimensions.workgroup_size() <= 256,
                "wg {} > 256 for size {size}",
                plan.dimensions.workgroup_size(),
            );
        }
    }

    #[test]
    fn property_global_always_gte_problem_size() {
        let planner = default_planner();
        for size in (1..=100).chain([500, 1000, 5000, 10000].iter().copied()) {
            let plan = planner.plan_1d(size);
            assert!(plan.dimensions.global_work_size[0] >= size);
        }
    }

    #[test]
    fn property_local_divides_global() {
        let planner = default_planner();
        for &size in &[1, 17, 64, 255, 1024, 9999] {
            let plan = planner.plan_1d(size);
            assert!(plan.dimensions.is_valid());
        }
    }

    // -----------------------------------------------------------------------
    // Round-up helper
    // -----------------------------------------------------------------------

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(0, 16), 0);
        assert_eq!(round_up(1, 16), 16);
        assert_eq!(round_up(16, 16), 16);
        assert_eq!(round_up(17, 16), 32);
        assert_eq!(round_up(100, 32), 128);
    }

    #[test]
    fn round_up_zero_multiple() {
        assert_eq!(round_up(42, 0), 42);
    }

    // -----------------------------------------------------------------------
    // largest_aligned_factor helper
    // -----------------------------------------------------------------------

    #[test]
    fn aligned_factor_basic() {
        assert_eq!(largest_aligned_factor(256, 256, 32), 256);
        assert_eq!(largest_aligned_factor(256, 128, 32), 128);
        assert_eq!(largest_aligned_factor(256, 1024, 32), 256);
    }

    #[test]
    fn aligned_factor_non_power_of_two() {
        // 1000 = 8*125, so largest multiple of 8 dividing 1000 and ≤512
        // is 8*62 = 496? No, 504=8*63. 1000/504 is not integer.
        // 1000/8 = 125. largest k*8 ≤ 512 where k*8 divides 1000.
        // Factors: 8,40,200,1000. ≤512: 200.
        assert_eq!(largest_aligned_factor(1000, 512, 8), 200);
    }

    #[test]
    fn aligned_factor_zero_n() {
        assert_eq!(largest_aligned_factor(0, 512, 32), 1);
    }
}
