//! Intel Arc A770 subgroup (SIMD lane) operations for OpenCL.
//!
//! Subgroup intrinsics allow efficient intra-workgroup communication
//! without shared local memory barriers. On Intel Xe-HPG (A770) the
//! hardware supports subgroup sizes of 8, 16, and 32 lanes. These
//! operations are critical for reduction and attention kernels.
//!
//! # Operations
//!
//! | Operation    | Description                                     |
//! |--------------|-------------------------------------------------|
//! | Shuffle      | Read another lane's value at a fixed offset      |
//! | Broadcast    | Copy one lane's value to every lane              |
//! | ReduceAdd    | Sum across all active lanes                      |
//! | ReduceMax    | Max across all active lanes                      |
//! | ReduceMin    | Min across all active lanes                      |
//! | ScanAdd      | Inclusive prefix sum                              |
//! | Ballot       | Bit-mask of lanes where predicate is true        |
//!
//! # CPU reference
//!
//! All public functions have pure-CPU scalar implementations so results
//! are deterministic and easy to validate against the OpenCL GPU path.

use std::fmt;

// ---------------------------------------------------------------------------
// SubgroupSize — supported SIMD widths
// ---------------------------------------------------------------------------

/// Subgroup (SIMD lane) width supported by the hardware.
///
/// Intel Arc A770 Xe-HPG supports all three sizes; the driver selects
/// a default but kernels may request a specific width via the
/// `intel_reqd_sub_group_size` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubgroupSize {
    /// 8 lanes — useful for small reductions and low-occupancy kernels.
    S8 = 8,
    /// 16 lanes — the most common default on Xe-HPG.
    S16 = 16,
    /// 32 lanes — maximum width, best for large reductions.
    S32 = 32,
}

impl SubgroupSize {
    /// Number of lanes as a `u32`.
    pub fn lanes(self) -> u32 {
        self as u32
    }

    /// Try to convert an integer to a [`SubgroupSize`].
    pub fn from_lanes(n: u32) -> Option<Self> {
        match n {
            8 => Some(Self::S8),
            16 => Some(Self::S16),
            32 => Some(Self::S32),
            _ => None,
        }
    }

    /// All supported subgroup sizes in ascending order.
    pub fn all() -> &'static [SubgroupSize] {
        &[Self::S8, Self::S16, Self::S32]
    }
}

impl fmt::Display for SubgroupSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sg{}", self.lanes())
    }
}

// ---------------------------------------------------------------------------
// SubgroupOp — operation enum
// ---------------------------------------------------------------------------

/// A subgroup-level intrinsic operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupOp {
    /// Cross-lane shuffle: read the value from `(lane_id + offset) % sg_size`.
    Shuffle(u32),
    /// Broadcast lane `lane`'s value to all lanes.
    Broadcast(u32),
    /// Sum-reduction across all active lanes.
    ReduceAdd,
    /// Max-reduction across all active lanes.
    ReduceMax,
    /// Min-reduction across all active lanes.
    ReduceMin,
    /// Inclusive prefix sum (each lane gets the sum of lanes 0..=self).
    ScanAdd,
    /// Ballot — bit-mask of lanes where the input predicate is non-zero.
    Ballot,
}

impl fmt::Display for SubgroupOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shuffle(o) => write!(f, "shuffle({o})"),
            Self::Broadcast(l) => write!(f, "broadcast({l})"),
            Self::ReduceAdd => write!(f, "reduce_add"),
            Self::ReduceMax => write!(f, "reduce_max"),
            Self::ReduceMin => write!(f, "reduce_min"),
            Self::ScanAdd => write!(f, "scan_add"),
            Self::Ballot => write!(f, "ballot"),
        }
    }
}

// ---------------------------------------------------------------------------
// SubgroupConfig
// ---------------------------------------------------------------------------

/// Runtime configuration for subgroup dispatching.
#[derive(Debug, Clone)]
pub struct SubgroupConfig {
    /// Preferred subgroup size for the device.
    pub preferred_size: SubgroupSize,
    /// Fallback size if the preferred one is not available.
    pub fallback_size: SubgroupSize,
    /// Enable cross-lane shuffle path (set `false` to force tree reduction).
    pub use_shuffle: bool,
}

impl SubgroupConfig {
    /// Create a new configuration.
    pub fn new(preferred: SubgroupSize, fallback: SubgroupSize, use_shuffle: bool) -> Self {
        Self { preferred_size: preferred, fallback_size: fallback, use_shuffle }
    }

    /// Default A770 config: prefer SG16, fallback SG8, shuffle enabled.
    pub fn a770_default() -> Self {
        Self::new(SubgroupSize::S16, SubgroupSize::S8, true)
    }

    /// Active subgroup size (preferred if available in a given list).
    pub fn active_size(&self, available: &[SubgroupSize]) -> SubgroupSize {
        if available.contains(&self.preferred_size) {
            self.preferred_size
        } else if available.contains(&self.fallback_size) {
            self.fallback_size
        } else {
            // Last resort: pick the first available.
            available.first().copied().unwrap_or(SubgroupSize::S16)
        }
    }
}

impl Default for SubgroupConfig {
    fn default() -> Self {
        Self::a770_default()
    }
}

// ---------------------------------------------------------------------------
// SubgroupReducer — intra-subgroup tree reduction
// ---------------------------------------------------------------------------

/// Performs tree reduction within a single subgroup.
///
/// The CPU reference implementation simulates the subgroup by treating
/// a contiguous slice as the set of lane values.
#[derive(Debug, Clone)]
pub struct SubgroupReducer {
    /// The subgroup width to simulate.
    pub size: SubgroupSize,
}

impl SubgroupReducer {
    pub fn new(size: SubgroupSize) -> Self {
        Self { size }
    }

    /// Sum-reduce `data[..size]`, returning the scalar result.
    ///
    /// If `data.len() < size.lanes()` (partial subgroup), only the
    /// provided lanes participate.
    pub fn reduce_add(&self, data: &[f32]) -> f32 {
        let n = data.len().min(self.size.lanes() as usize);
        data[..n].iter().sum()
    }

    /// Max-reduce `data[..size]`.
    pub fn reduce_max(&self, data: &[f32]) -> f32 {
        let n = data.len().min(self.size.lanes() as usize);
        data[..n].iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Min-reduce `data[..size]`.
    pub fn reduce_min(&self, data: &[f32]) -> f32 {
        let n = data.len().min(self.size.lanes() as usize);
        data[..n].iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Inclusive prefix sum. Returns a `Vec` of the same length as `data`
    /// (clamped to subgroup size).
    pub fn inclusive_scan_add(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len().min(self.size.lanes() as usize);
        let mut out = Vec::with_capacity(n);
        let mut acc = 0.0f32;
        for &v in &data[..n] {
            acc += v;
            out.push(acc);
        }
        out
    }

    /// Exclusive prefix sum (identity = 0). Returns a `Vec` of the same
    /// length as `data` (clamped to subgroup size).
    pub fn exclusive_scan_add(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len().min(self.size.lanes() as usize);
        let mut out = Vec::with_capacity(n);
        let mut acc = 0.0f32;
        for &v in &data[..n] {
            out.push(acc);
            acc += v;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// SubgroupShuffle — cross-lane data exchange
// ---------------------------------------------------------------------------

/// CPU reference for subgroup shuffle (cross-lane reads).
#[derive(Debug, Clone)]
pub struct SubgroupShuffle {
    pub size: SubgroupSize,
}

impl SubgroupShuffle {
    pub fn new(size: SubgroupSize) -> Self {
        Self { size }
    }

    /// Shuffle with a fixed offset: lane `i` reads from
    /// `(i + offset) % subgroup_size`.
    ///
    /// Returns a new vector of the same length as `data` (clamped to
    /// subgroup size).
    pub fn shuffle(&self, data: &[f32], offset: u32) -> Vec<f32> {
        let n = data.len().min(self.size.lanes() as usize);
        (0..n)
            .map(|i| {
                let src = (i + offset as usize) % n;
                data[src]
            })
            .collect()
    }

    /// Broadcast lane `src_lane`'s value to all lanes.
    pub fn broadcast(&self, data: &[f32], src_lane: u32) -> Vec<f32> {
        let n = data.len().min(self.size.lanes() as usize);
        let src = (src_lane as usize).min(n.saturating_sub(1));
        vec![data[src]; n]
    }

    /// Butterfly (XOR) shuffle: lane `i` reads from `i ^ mask`.
    pub fn shuffle_xor(&self, data: &[f32], mask: u32) -> Vec<f32> {
        let n = data.len().min(self.size.lanes() as usize);
        (0..n)
            .map(|i| {
                let src = i ^ (mask as usize);
                if src < n { data[src] } else { data[i] }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SubgroupBallot — lane predicate mask
// ---------------------------------------------------------------------------

/// CPU reference for subgroup ballot.
#[derive(Debug, Clone)]
pub struct SubgroupBallot {
    pub size: SubgroupSize,
}

/// Result of a ballot operation — a bit-mask of active lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BallotResult {
    /// Bit `i` is set if lane `i` had a non-zero predicate.
    pub mask: u64,
    /// Number of lanes that participated.
    pub active_lanes: u32,
}

impl BallotResult {
    /// Number of lanes with a set bit (popcount).
    pub fn count_ones(&self) -> u32 {
        self.mask.count_ones()
    }

    /// Whether lane `i` is set.
    pub fn is_set(&self, lane: u32) -> bool {
        lane < 64 && (self.mask >> lane) & 1 == 1
    }
}

impl SubgroupBallot {
    pub fn new(size: SubgroupSize) -> Self {
        Self { size }
    }

    /// Compute ballot: bit `i` is set when `predicates[i] != 0`.
    pub fn ballot(&self, predicates: &[i32]) -> BallotResult {
        let n = predicates.len().min(self.size.lanes() as usize);
        let mut mask = 0u64;
        for (i, &p) in predicates[..n].iter().enumerate() {
            if p != 0 {
                mask |= 1u64 << i;
            }
        }
        BallotResult { mask, active_lanes: n as u32 }
    }
}

// ---------------------------------------------------------------------------
// WorkgroupReducer — multi-subgroup hierarchical reduction
// ---------------------------------------------------------------------------

/// Hierarchical workgroup reduction: each subgroup reduces internally,
/// then partial results are combined across subgroups.
#[derive(Debug, Clone)]
pub struct WorkgroupReducer {
    /// Subgroup size used for intra-subgroup reduction.
    pub subgroup_size: SubgroupSize,
}

impl WorkgroupReducer {
    pub fn new(subgroup_size: SubgroupSize) -> Self {
        Self { subgroup_size }
    }

    /// Number of subgroups needed for `workgroup_size` lanes.
    pub fn num_subgroups(&self, workgroup_size: usize) -> usize {
        let sg = self.subgroup_size.lanes() as usize;
        workgroup_size.div_ceil(sg)
    }

    /// Sum-reduce an entire workgroup of values.
    pub fn reduce_add(&self, data: &[f32]) -> f32 {
        let sg = self.subgroup_size.lanes() as usize;
        let reducer = SubgroupReducer::new(self.subgroup_size);

        // Phase 1: reduce within each subgroup.
        let mut partials: Vec<f32> = Vec::new();
        for chunk in data.chunks(sg) {
            partials.push(reducer.reduce_add(chunk));
        }

        // Phase 2: reduce the partial results (may itself span subgroups).
        if partials.len() <= sg {
            reducer.reduce_add(&partials)
        } else {
            // Recursive — rare for realistic workgroup sizes.
            self.reduce_add(&partials)
        }
    }

    /// Max-reduce an entire workgroup.
    pub fn reduce_max(&self, data: &[f32]) -> f32 {
        let sg = self.subgroup_size.lanes() as usize;
        let reducer = SubgroupReducer::new(self.subgroup_size);

        let mut partials: Vec<f32> = Vec::new();
        for chunk in data.chunks(sg) {
            partials.push(reducer.reduce_max(chunk));
        }

        if partials.len() <= sg {
            reducer.reduce_max(&partials)
        } else {
            self.reduce_max(&partials)
        }
    }

    /// Min-reduce an entire workgroup.
    pub fn reduce_min(&self, data: &[f32]) -> f32 {
        let sg = self.subgroup_size.lanes() as usize;
        let reducer = SubgroupReducer::new(self.subgroup_size);

        let mut partials: Vec<f32> = Vec::new();
        for chunk in data.chunks(sg) {
            partials.push(reducer.reduce_min(chunk));
        }

        if partials.len() <= sg {
            reducer.reduce_min(&partials)
        } else {
            self.reduce_min(&partials)
        }
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel sources — sub_group intrinsics
// ---------------------------------------------------------------------------

/// OpenCL kernel source using `cl_khr_subgroups` / `cl_intel_subgroups`.
///
/// These kernels rely on hardware subgroup intrinsics and must be compiled
/// with `-cl-std=CL2.0` (or newer) on a driver that exposes the extension.
pub fn subgroup_reduce_kernel_source() -> &'static str {
    r#"
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

// ── Subgroup reduce-add ────────────────────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_reduce_add(
    __global const float* restrict input,
    __global float* restrict output,
    const uint n)
{
    uint gid = get_global_id(0);
    float val = (gid < n) ? input[gid] : 0.0f;
    float sg_sum = sub_group_reduce_add(val);
    if (get_sub_group_local_id() == 0) {
        uint sg_id = get_sub_group_id()
                   + get_group_id(0) * get_num_sub_groups();
        output[sg_id] = sg_sum;
    }
}

// ── Subgroup reduce-max ────────────────────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_reduce_max(
    __global const float* restrict input,
    __global float* restrict output,
    const uint n)
{
    uint gid = get_global_id(0);
    float val = (gid < n) ? input[gid] : -INFINITY;
    float sg_max = sub_group_reduce_max(val);
    if (get_sub_group_local_id() == 0) {
        uint sg_id = get_sub_group_id()
                   + get_group_id(0) * get_num_sub_groups();
        output[sg_id] = sg_max;
    }
}

// ── Subgroup reduce-min ────────────────────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_reduce_min(
    __global const float* restrict input,
    __global float* restrict output,
    const uint n)
{
    uint gid = get_global_id(0);
    float val = (gid < n) ? input[gid] : INFINITY;
    float sg_min = sub_group_reduce_min(val);
    if (get_sub_group_local_id() == 0) {
        uint sg_id = get_sub_group_id()
                   + get_group_id(0) * get_num_sub_groups();
        output[sg_id] = sg_min;
    }
}

// ── Subgroup broadcast ─────────────────────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_broadcast(
    __global const float* restrict input,
    __global float* restrict output,
    const uint src_lane,
    const uint n)
{
    uint gid = get_global_id(0);
    float val = (gid < n) ? input[gid] : 0.0f;
    float bcast = sub_group_broadcast(val, src_lane);
    if (gid < n) {
        output[gid] = bcast;
    }
}

// ── Subgroup shuffle ───────────────────────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_shuffle(
    __global const float* restrict input,
    __global float* restrict output,
    const uint offset,
    const uint n)
{
    uint gid = get_global_id(0);
    float val = (gid < n) ? input[gid] : 0.0f;
    uint src = (get_sub_group_local_id() + offset) % get_sub_group_size();
    float shuffled = intel_sub_group_shuffle(val, src);
    if (gid < n) {
        output[gid] = shuffled;
    }
}

// ── Subgroup inclusive scan-add ─────────────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_scan_add(
    __global const float* restrict input,
    __global float* restrict output,
    const uint n)
{
    uint gid = get_global_id(0);
    float val = (gid < n) ? input[gid] : 0.0f;
    float scan = sub_group_scan_inclusive_add(val);
    if (gid < n) {
        output[gid] = scan;
    }
}

// ── Workgroup hierarchical reduce-add ───────────────────────────────────
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void workgroup_reduce_add(
    __global const float* restrict input,
    __global float* restrict output,
    __local float* scratch,
    const uint n)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    float val = (gid < n) ? input[gid] : 0.0f;
    float sg_sum = sub_group_reduce_add(val);

    // First lane of each subgroup writes to SLM.
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sg_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // First subgroup reduces the partial sums.
    if (get_sub_group_id() == 0) {
        float partial = (lid < get_num_sub_groups())
                      ? scratch[lid] : 0.0f;
        float total = sub_group_reduce_add(partial);
        if (lid == 0) {
            output[get_group_id(0)] = total;
        }
    }
}
"#
}

/// Returns the ballot kernel source (requires `cl_khr_subgroup_ballot`).
pub fn subgroup_ballot_kernel_source() -> &'static str {
    r#"
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_ballot : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subgroup_ballot(
    __global const int* restrict predicates,
    __global ulong* restrict ballot_out,
    const uint n)
{
    uint gid = get_global_id(0);
    int pred = (gid < n) ? predicates[gid] : 0;
    ulong mask = sub_group_ballot(pred != 0);
    if (get_sub_group_local_id() == 0) {
        uint sg_id = get_sub_group_id()
                   + get_group_id(0) * get_num_sub_groups();
        ballot_out[sg_id] = mask;
    }
}
"#
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SubgroupSize ────────────────────────────────────────────────────

    #[test]
    fn subgroup_size_lanes() {
        assert_eq!(SubgroupSize::S8.lanes(), 8);
        assert_eq!(SubgroupSize::S16.lanes(), 16);
        assert_eq!(SubgroupSize::S32.lanes(), 32);
    }

    #[test]
    fn subgroup_size_from_lanes_valid() {
        assert_eq!(SubgroupSize::from_lanes(8), Some(SubgroupSize::S8));
        assert_eq!(SubgroupSize::from_lanes(16), Some(SubgroupSize::S16));
        assert_eq!(SubgroupSize::from_lanes(32), Some(SubgroupSize::S32));
    }

    #[test]
    fn subgroup_size_from_lanes_invalid() {
        assert_eq!(SubgroupSize::from_lanes(0), None);
        assert_eq!(SubgroupSize::from_lanes(4), None);
        assert_eq!(SubgroupSize::from_lanes(64), None);
    }

    #[test]
    fn subgroup_size_display() {
        assert_eq!(format!("{}", SubgroupSize::S8), "sg8");
        assert_eq!(format!("{}", SubgroupSize::S16), "sg16");
        assert_eq!(format!("{}", SubgroupSize::S32), "sg32");
    }

    #[test]
    fn subgroup_size_all() {
        let all = SubgroupSize::all();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], SubgroupSize::S8);
        assert_eq!(all[2], SubgroupSize::S32);
    }

    // ── SubgroupOp ──────────────────────────────────────────────────────

    #[test]
    fn subgroup_op_display() {
        assert_eq!(format!("{}", SubgroupOp::Shuffle(3)), "shuffle(3)");
        assert_eq!(format!("{}", SubgroupOp::Broadcast(0)), "broadcast(0)");
        assert_eq!(format!("{}", SubgroupOp::ReduceAdd), "reduce_add");
        assert_eq!(format!("{}", SubgroupOp::ReduceMax), "reduce_max");
        assert_eq!(format!("{}", SubgroupOp::ReduceMin), "reduce_min");
        assert_eq!(format!("{}", SubgroupOp::ScanAdd), "scan_add");
        assert_eq!(format!("{}", SubgroupOp::Ballot), "ballot");
    }

    #[test]
    fn subgroup_op_equality() {
        assert_eq!(SubgroupOp::Shuffle(1), SubgroupOp::Shuffle(1));
        assert_ne!(SubgroupOp::Shuffle(1), SubgroupOp::Shuffle(2));
        assert_ne!(SubgroupOp::ReduceAdd, SubgroupOp::ReduceMax);
    }

    // ── SubgroupConfig ──────────────────────────────────────────────────

    #[test]
    fn config_a770_default() {
        let cfg = SubgroupConfig::a770_default();
        assert_eq!(cfg.preferred_size, SubgroupSize::S16);
        assert_eq!(cfg.fallback_size, SubgroupSize::S8);
        assert!(cfg.use_shuffle);
    }

    #[test]
    fn config_active_size_preferred_available() {
        let cfg = SubgroupConfig::new(SubgroupSize::S32, SubgroupSize::S16, true);
        let avail = [SubgroupSize::S8, SubgroupSize::S16, SubgroupSize::S32];
        assert_eq!(cfg.active_size(&avail), SubgroupSize::S32);
    }

    #[test]
    fn config_active_size_falls_back() {
        let cfg = SubgroupConfig::new(SubgroupSize::S32, SubgroupSize::S16, true);
        let avail = [SubgroupSize::S8, SubgroupSize::S16];
        assert_eq!(cfg.active_size(&avail), SubgroupSize::S16);
    }

    #[test]
    fn config_active_size_last_resort() {
        let cfg = SubgroupConfig::new(SubgroupSize::S32, SubgroupSize::S16, true);
        let avail = [SubgroupSize::S8];
        assert_eq!(cfg.active_size(&avail), SubgroupSize::S8);
    }

    #[test]
    fn config_active_size_empty_list() {
        let cfg = SubgroupConfig::default();
        let avail: &[SubgroupSize] = &[];
        // Falls back to S16 when nothing available.
        assert_eq!(cfg.active_size(avail), SubgroupSize::S16);
    }

    // ── SubgroupReducer — reduce_add ────────────────────────────────────

    #[test]
    fn reduce_add_s8_full() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        assert_eq!(r.reduce_add(&data), 36.0);
    }

    #[test]
    fn reduce_add_s16_full() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        assert_eq!(r.reduce_add(&data), 136.0);
    }

    #[test]
    fn reduce_add_s32_full() {
        let r = SubgroupReducer::new(SubgroupSize::S32);
        let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        assert_eq!(r.reduce_add(&data), 528.0);
    }

    #[test]
    fn reduce_add_partial_subgroup() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data = vec![1.0, 2.0, 3.0]; // only 3 of 16 lanes
        assert_eq!(r.reduce_add(&data), 6.0);
    }

    #[test]
    fn reduce_add_single_lane() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        assert_eq!(r.reduce_add(&[42.0]), 42.0);
    }

    #[test]
    fn reduce_add_empty() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        assert_eq!(r.reduce_add(&[]), 0.0);
    }

    // ── SubgroupReducer — reduce_max ────────────────────────────────────

    #[test]
    fn reduce_max_s8() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        let data = vec![-3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 0.0];
        assert_eq!(r.reduce_max(&data), 9.0);
    }

    #[test]
    fn reduce_max_s16() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data: Vec<f32> = (0..16).map(|x| -(x as f32)).collect();
        assert_eq!(r.reduce_max(&data), 0.0);
    }

    #[test]
    fn reduce_max_s32() {
        let r = SubgroupReducer::new(SubgroupSize::S32);
        let mut data: Vec<f32> = vec![0.0; 32];
        data[31] = 100.0;
        assert_eq!(r.reduce_max(&data), 100.0);
    }

    #[test]
    fn reduce_max_partial() {
        let r = SubgroupReducer::new(SubgroupSize::S32);
        let data = vec![5.0, 3.0];
        assert_eq!(r.reduce_max(&data), 5.0);
    }

    #[test]
    fn reduce_max_single_lane() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        assert_eq!(r.reduce_max(&[-7.0]), -7.0);
    }

    // ── SubgroupReducer — reduce_min ────────────────────────────────────

    #[test]
    fn reduce_min_s8() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        let data = vec![3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 0.5, 8.0];
        assert_eq!(r.reduce_min(&data), 0.5);
    }

    #[test]
    fn reduce_min_s16() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        assert_eq!(r.reduce_min(&data), 1.0);
    }

    #[test]
    fn reduce_min_s32() {
        let r = SubgroupReducer::new(SubgroupSize::S32);
        let mut data: Vec<f32> = vec![100.0; 32];
        data[15] = -42.0;
        assert_eq!(r.reduce_min(&data), -42.0);
    }

    // ── SubgroupReducer — inclusive scan ─────────────────────────────────

    #[test]
    fn inclusive_scan_s8() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let scan = r.inclusive_scan_add(&data);
        assert_eq!(scan, vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]);
    }

    #[test]
    fn inclusive_scan_s16() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data = vec![1.0; 16];
        let scan = r.inclusive_scan_add(&data);
        let expected: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        assert_eq!(scan, expected);
    }

    #[test]
    fn inclusive_scan_s32() {
        let r = SubgroupReducer::new(SubgroupSize::S32);
        let data = vec![2.0; 32];
        let scan = r.inclusive_scan_add(&data);
        assert_eq!(scan.len(), 32);
        assert_eq!(scan[0], 2.0);
        assert_eq!(scan[31], 64.0);
    }

    #[test]
    fn inclusive_scan_partial() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data = vec![1.0, 2.0, 3.0];
        let scan = r.inclusive_scan_add(&data);
        assert_eq!(scan, vec![1.0, 3.0, 6.0]);
    }

    // ── SubgroupReducer — exclusive scan ────────────────────────────────

    #[test]
    fn exclusive_scan_s8() {
        let r = SubgroupReducer::new(SubgroupSize::S8);
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let scan = r.exclusive_scan_add(&data);
        assert_eq!(scan, vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    }

    #[test]
    fn exclusive_scan_s16() {
        let r = SubgroupReducer::new(SubgroupSize::S16);
        let data = vec![1.0; 16];
        let scan = r.exclusive_scan_add(&data);
        let expected: Vec<f32> = (0..16).map(|x| x as f32).collect();
        assert_eq!(scan, expected);
    }

    #[test]
    fn exclusive_scan_partial() {
        let r = SubgroupReducer::new(SubgroupSize::S32);
        let data = vec![3.0, 5.0];
        let scan = r.exclusive_scan_add(&data);
        assert_eq!(scan, vec![0.0, 3.0]);
    }

    // ── SubgroupShuffle ─────────────────────────────────────────────────

    #[test]
    fn shuffle_offset_s8() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let result = s.shuffle(&data, 1);
        // lane i reads from (i+1)%8
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]);
    }

    #[test]
    fn shuffle_offset_s16() {
        let s = SubgroupShuffle::new(SubgroupSize::S16);
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let result = s.shuffle(&data, 4);
        assert_eq!(result[0], 4.0);
        assert_eq!(result[12], 0.0); // (12+4)%16 = 0
    }

    #[test]
    fn shuffle_offset_zero() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data: Vec<f32> = (10..18).map(|x| x as f32).collect();
        let result = s.shuffle(&data, 0);
        assert_eq!(result, data);
    }

    #[test]
    fn shuffle_offset_wrap() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        // offset = size → wraps to same position
        let result = s.shuffle(&data, 8);
        assert_eq!(result, data);
    }

    #[test]
    fn shuffle_partial_subgroup() {
        let s = SubgroupShuffle::new(SubgroupSize::S16);
        let data = vec![10.0, 20.0, 30.0];
        let result = s.shuffle(&data, 1);
        assert_eq!(result, vec![20.0, 30.0, 10.0]);
    }

    // ── SubgroupShuffle — broadcast ─────────────────────────────────────

    #[test]
    fn broadcast_lane_0_s8() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let result = s.broadcast(&data, 0);
        assert_eq!(result, vec![0.0; 8]);
    }

    #[test]
    fn broadcast_lane_middle_s16() {
        let s = SubgroupShuffle::new(SubgroupSize::S16);
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let result = s.broadcast(&data, 7);
        assert_eq!(result, vec![7.0; 16]);
    }

    #[test]
    fn broadcast_last_lane_s32() {
        let s = SubgroupShuffle::new(SubgroupSize::S32);
        let data: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let result = s.broadcast(&data, 31);
        assert_eq!(result, vec![31.0; 32]);
    }

    #[test]
    fn broadcast_partial_subgroup() {
        let s = SubgroupShuffle::new(SubgroupSize::S16);
        let data = vec![100.0, 200.0, 300.0];
        let result = s.broadcast(&data, 1);
        assert_eq!(result, vec![200.0; 3]);
    }

    #[test]
    fn broadcast_out_of_range_clamps() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data = vec![1.0, 2.0, 3.0];
        // lane 99 clamps to last valid lane (2)
        let result = s.broadcast(&data, 99);
        assert_eq!(result, vec![3.0; 3]);
    }

    // ── SubgroupShuffle — XOR ───────────────────────────────────────────

    #[test]
    fn shuffle_xor_s8_mask1() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let result = s.shuffle_xor(&data, 1);
        // lane 0 reads 1, lane 1 reads 0, lane 2 reads 3, lane 3 reads 2, ...
        assert_eq!(result, vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]);
    }

    #[test]
    fn shuffle_xor_s16_mask2() {
        let s = SubgroupShuffle::new(SubgroupSize::S16);
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let result = s.shuffle_xor(&data, 2);
        assert_eq!(result[0], 2.0); // 0^2 = 2
        assert_eq!(result[2], 0.0); // 2^2 = 0
        assert_eq!(result[5], 7.0); // 5^2 = 7
    }

    #[test]
    fn shuffle_xor_identity() {
        let s = SubgroupShuffle::new(SubgroupSize::S8);
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let result = s.shuffle_xor(&data, 0);
        assert_eq!(result, data);
    }

    // ── SubgroupBallot ──────────────────────────────────────────────────

    #[test]
    fn ballot_all_true_s8() {
        let b = SubgroupBallot::new(SubgroupSize::S8);
        let preds = vec![1; 8];
        let result = b.ballot(&preds);
        assert_eq!(result.mask, 0xFF);
        assert_eq!(result.count_ones(), 8);
        assert_eq!(result.active_lanes, 8);
    }

    #[test]
    fn ballot_all_false_s16() {
        let b = SubgroupBallot::new(SubgroupSize::S16);
        let preds = vec![0; 16];
        let result = b.ballot(&preds);
        assert_eq!(result.mask, 0);
        assert_eq!(result.count_ones(), 0);
    }

    #[test]
    fn ballot_alternating_s16() {
        let b = SubgroupBallot::new(SubgroupSize::S16);
        let preds: Vec<i32> = (0..16).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let result = b.ballot(&preds);
        assert_eq!(result.mask, 0x5555);
        assert_eq!(result.count_ones(), 8);
    }

    #[test]
    fn ballot_s32_sparse() {
        let b = SubgroupBallot::new(SubgroupSize::S32);
        let mut preds = vec![0i32; 32];
        preds[0] = 1;
        preds[31] = 1;
        let result = b.ballot(&preds);
        assert!(result.is_set(0));
        assert!(!result.is_set(1));
        assert!(result.is_set(31));
        assert_eq!(result.count_ones(), 2);
    }

    #[test]
    fn ballot_partial_subgroup() {
        let b = SubgroupBallot::new(SubgroupSize::S16);
        let preds = vec![1, 0, 1]; // only 3 of 16 lanes
        let result = b.ballot(&preds);
        assert_eq!(result.mask, 0b101);
        assert_eq!(result.active_lanes, 3);
    }

    #[test]
    fn ballot_is_set_bounds() {
        let result = BallotResult { mask: 0xFF, active_lanes: 8 };
        assert!(result.is_set(0));
        assert!(result.is_set(7));
        assert!(!result.is_set(8));
        // lane 64+ always false (u64 range)
        assert!(!result.is_set(64));
    }

    // ── WorkgroupReducer ────────────────────────────────────────────────

    #[test]
    fn workgroup_reduce_add_single_subgroup() {
        let wr = WorkgroupReducer::new(SubgroupSize::S8);
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        assert_eq!(wr.reduce_add(&data), 36.0);
    }

    #[test]
    fn workgroup_reduce_add_two_subgroups() {
        let wr = WorkgroupReducer::new(SubgroupSize::S8);
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        // Two SG8 subgroups: 1..=8 sum=36, 9..=16 sum=100 → total=136
        assert_eq!(wr.reduce_add(&data), 136.0);
    }

    #[test]
    fn workgroup_reduce_add_four_subgroups() {
        let wr = WorkgroupReducer::new(SubgroupSize::S16);
        let data = vec![1.0; 64]; // 4 subgroups of SG16
        assert_eq!(wr.reduce_add(&data), 64.0);
    }

    #[test]
    fn workgroup_reduce_add_partial_last_subgroup() {
        let wr = WorkgroupReducer::new(SubgroupSize::S8);
        let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        // SG0: 1..=8 = 36, SG1: 9+10 = 19 → 55
        assert_eq!(wr.reduce_add(&data), 55.0);
    }

    #[test]
    fn workgroup_reduce_max_multi() {
        let wr = WorkgroupReducer::new(SubgroupSize::S8);
        let mut data = vec![-1.0; 24]; // 3 subgroups
        data[20] = 99.0;
        assert_eq!(wr.reduce_max(&data), 99.0);
    }

    #[test]
    fn workgroup_reduce_min_multi() {
        let wr = WorkgroupReducer::new(SubgroupSize::S16);
        let mut data = vec![50.0; 48]; // 3 subgroups
        data[33] = -7.0;
        assert_eq!(wr.reduce_min(&data), -7.0);
    }

    #[test]
    fn workgroup_num_subgroups() {
        let wr = WorkgroupReducer::new(SubgroupSize::S16);
        assert_eq!(wr.num_subgroups(16), 1);
        assert_eq!(wr.num_subgroups(17), 2);
        assert_eq!(wr.num_subgroups(256), 16);
    }

    #[test]
    fn workgroup_reduce_add_empty() {
        let wr = WorkgroupReducer::new(SubgroupSize::S8);
        assert_eq!(wr.reduce_add(&[]), 0.0);
    }

    // ── Property tests ──────────────────────────────────────────────────

    #[test]
    fn property_reduce_add_equals_sequential_sum() {
        for &size in SubgroupSize::all() {
            let r = SubgroupReducer::new(size);
            let n = size.lanes() as usize;
            let data: Vec<f32> = (0..n).map(|x| (x as f32) * 0.5).collect();
            let expected: f32 = data.iter().sum();
            let result = r.reduce_add(&data);
            assert!(
                (result - expected).abs() < 1e-5,
                "reduce_add mismatch for {size}: got {result}, expected {expected}",
            );
        }
    }

    #[test]
    fn property_reduce_max_equals_sequential_max() {
        for &size in SubgroupSize::all() {
            let r = SubgroupReducer::new(size);
            let n = size.lanes() as usize;
            let data: Vec<f32> = (0..n).map(|x| (x as f32) - (n as f32 / 2.0)).collect();
            let expected = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let result = r.reduce_max(&data);
            assert_eq!(result, expected, "reduce_max mismatch for {size}");
        }
    }

    #[test]
    fn property_reduce_min_equals_sequential_min() {
        for &size in SubgroupSize::all() {
            let r = SubgroupReducer::new(size);
            let n = size.lanes() as usize;
            let data: Vec<f32> = (0..n).map(|x| (x as f32) - (n as f32 / 2.0)).collect();
            let expected = data.iter().copied().fold(f32::INFINITY, f32::min);
            let result = r.reduce_min(&data);
            assert_eq!(result, expected, "reduce_min mismatch for {size}");
        }
    }

    #[test]
    fn property_inclusive_scan_last_equals_reduce_add() {
        for &size in SubgroupSize::all() {
            let r = SubgroupReducer::new(size);
            let n = size.lanes() as usize;
            let data: Vec<f32> = (1..=n).map(|x| x as f32).collect();
            let scan = r.inclusive_scan_add(&data);
            let total = r.reduce_add(&data);
            assert!((scan[n - 1] - total).abs() < 1e-4, "scan last != reduce_add for {size}");
        }
    }

    #[test]
    fn property_exclusive_scan_shift() {
        for &size in SubgroupSize::all() {
            let r = SubgroupReducer::new(size);
            let n = size.lanes() as usize;
            let data: Vec<f32> = (1..=n).map(|x| x as f32).collect();
            let inc = r.inclusive_scan_add(&data);
            let exc = r.exclusive_scan_add(&data);
            // exclusive[i] = inclusive[i] - data[i]
            for i in 0..n {
                let diff = (exc[i] - (inc[i] - data[i])).abs();
                assert!(diff < 1e-5, "scan shift mismatch at lane {i} for {size}");
            }
        }
    }

    #[test]
    fn property_workgroup_reduce_matches_flat_sum() {
        for &size in SubgroupSize::all() {
            let wr = WorkgroupReducer::new(size);
            let n = size.lanes() as usize * 3 + 5; // non-multiple
            let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
            let expected: f32 = data.iter().sum();
            let result = wr.reduce_add(&data);
            assert!(
                (result - expected).abs() < 1e-3,
                "workgroup reduce_add mismatch for {size}: {result} vs {expected}",
            );
        }
    }

    #[test]
    fn property_ballot_count_equals_nonzero_count() {
        for &size in SubgroupSize::all() {
            let b = SubgroupBallot::new(size);
            let n = size.lanes() as usize;
            // Every third lane is active.
            let preds: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
            let result = b.ballot(&preds);
            let expected = preds.iter().filter(|&&p| p != 0).count() as u32;
            assert_eq!(result.count_ones(), expected, "ballot count mismatch for {size}");
        }
    }

    // ── Kernel source smoke tests ───────────────────────────────────────

    #[test]
    fn kernel_source_contains_reduce_add() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("subgroup_reduce_add"));
        assert!(src.contains("sub_group_reduce_add"));
    }

    #[test]
    fn kernel_source_contains_reduce_max() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("subgroup_reduce_max"));
        assert!(src.contains("sub_group_reduce_max"));
    }

    #[test]
    fn kernel_source_contains_reduce_min() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("subgroup_reduce_min"));
        assert!(src.contains("sub_group_reduce_min"));
    }

    #[test]
    fn kernel_source_contains_broadcast() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("subgroup_broadcast"));
        assert!(src.contains("sub_group_broadcast"));
    }

    #[test]
    fn kernel_source_contains_shuffle() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("subgroup_shuffle"));
        assert!(src.contains("intel_sub_group_shuffle"));
    }

    #[test]
    fn kernel_source_contains_scan() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("subgroup_scan_add"));
        assert!(src.contains("sub_group_scan_inclusive_add"));
    }

    #[test]
    fn kernel_source_contains_workgroup_reduce() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("workgroup_reduce_add"));
        assert!(src.contains("__local"));
    }

    #[test]
    fn ballot_kernel_source_valid() {
        let src = subgroup_ballot_kernel_source();
        assert!(src.contains("subgroup_ballot"));
        assert!(src.contains("sub_group_ballot"));
        assert!(src.contains("cl_khr_subgroup_ballot"));
    }

    #[test]
    fn kernel_source_has_intel_reqd_subgroup_size() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("intel_reqd_sub_group_size"));
    }

    #[test]
    fn kernel_source_has_khr_subgroups_extension() {
        let src = subgroup_reduce_kernel_source();
        assert!(src.contains("cl_khr_subgroups"));
    }
}
