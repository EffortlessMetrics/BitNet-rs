//! CUDA warp shuffle primitives with CPU reference implementations.
//!
//! # Overview
//!
//! Warp shuffles allow threads within a single warp (typically 32 lanes) to
//! exchange register values without going through shared memory. This module
//! provides four shuffle variants, warp-level reductions, prefix scans, vote
//! functions, and broadcast — all with pure-Rust CPU reference
//! implementations for correctness testing on machines without a GPU.
//!
//! # Shuffle variants
//!
//! | Variant          | Description                              |
//! |------------------|------------------------------------------|
//! | `shfl_sync`      | Read from an arbitrary lane (indexed)    |
//! | `shfl_up_sync`   | Read from a lane with lower index        |
//! | `shfl_down_sync` | Read from a lane with higher index       |
//! | `shfl_xor_sync`  | Read from a lane with XOR'd index        |
//!
//! # GPU path
//!
//! On builds with `feature = "gpu"` or `feature = "cuda"`, the primitives
//! will be backed by real `__shfl_sync` PTX intrinsics (future work). The
//! CPU reference implementations serve as the canonical correctness baseline.
//!
//! # CPU fallback
//!
//! Every operation has a scalar CPU implementation that simulates the warp
//! behaviour over a plain `&[f32]` register file. These are always compiled.

use std::fmt;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during warp shuffle operations.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum WarpShuffleError {
    /// The source lane index is out of range for the configured warp size.
    #[error("invalid lane index {lane} for warp size {warp_size}")]
    InvalidLane { lane: u32, warp_size: u32 },

    /// The active mask has bits set beyond the warp size.
    #[error("invalid mask 0x{mask:08x} for warp size {warp_size}")]
    InvalidMask { mask: u32, warp_size: u32 },

    /// The warp size is not a power of two or exceeds 32.
    #[error("invalid warp size {0} (must be a power of two, 1..=32)")]
    InvalidWarpSize(u32),

    /// The lane width is not a power of two or exceeds the warp size.
    #[error("invalid lane width {lane_width} for warp size {warp_size}")]
    InvalidLaneWidth { lane_width: u32, warp_size: u32 },

    /// The register file length does not match the warp size.
    #[error("register file length {got} does not match warp size {expected}")]
    RegisterFileMismatch { expected: u32, got: usize },
}

type Result<T> = std::result::Result<T, WarpShuffleError>;

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Configuration describing a warp's geometry and active mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpConfig {
    /// Number of threads in the warp (must be a power of two, 1..=32).
    pub warp_size: u32,
    /// Bitmask of active lanes. Only bits `0..warp_size` may be set.
    pub active_mask: u32,
}

impl WarpConfig {
    /// Create a new configuration with full warp participation.
    ///
    /// # Errors
    ///
    /// Returns [`WarpShuffleError::InvalidWarpSize`] if `warp_size` is not a
    /// power of two in `1..=32`.
    pub fn new(warp_size: u32) -> Result<Self> {
        Self::validate_warp_size(warp_size)?;
        let active_mask = if warp_size == 32 { 0xFFFF_FFFF } else { (1u32 << warp_size) - 1 };
        Ok(Self { warp_size, active_mask })
    }

    /// Create a configuration with a custom active mask.
    ///
    /// # Errors
    ///
    /// Returns an error if `warp_size` is invalid or if `active_mask` has
    /// bits set beyond `warp_size`.
    pub fn with_mask(warp_size: u32, active_mask: u32) -> Result<Self> {
        Self::validate_warp_size(warp_size)?;
        let max_mask = if warp_size == 32 { 0xFFFF_FFFF } else { (1u32 << warp_size) - 1 };
        if active_mask & !max_mask != 0 {
            return Err(WarpShuffleError::InvalidMask { mask: active_mask, warp_size });
        }
        Ok(Self { warp_size, active_mask })
    }

    /// The default 32-lane warp with all lanes active.
    #[must_use]
    pub fn default_32() -> Self {
        Self { warp_size: 32, active_mask: 0xFFFF_FFFF }
    }

    /// Returns `true` if the given lane is active.
    #[must_use]
    pub fn is_active(&self, lane: u32) -> bool {
        lane < self.warp_size && (self.active_mask >> lane) & 1 == 1
    }

    /// Count the number of active lanes.
    #[must_use]
    pub fn active_count(&self) -> u32 {
        self.active_mask.count_ones()
    }

    fn validate_warp_size(warp_size: u32) -> Result<()> {
        if warp_size == 0 || warp_size > 32 || !warp_size.is_power_of_two() {
            return Err(WarpShuffleError::InvalidWarpSize(warp_size));
        }
        Ok(())
    }

    fn validate_lane_width(&self, lane_width: u32) -> Result<()> {
        if lane_width == 0 || !lane_width.is_power_of_two() || lane_width > self.warp_size {
            return Err(WarpShuffleError::InvalidLaneWidth {
                lane_width,
                warp_size: self.warp_size,
            });
        }
        Ok(())
    }
}

/// Specifies which shuffle variant to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleMode {
    /// Read from an arbitrary lane within the sub-warp segment.
    Indexed {
        /// Source lane index within the segment.
        src_lane: u32,
    },
    /// Read from a lane `delta` positions lower (towards lane 0).
    Up {
        /// Distance to shift up.
        delta: u32,
    },
    /// Read from a lane `delta` positions higher.
    Down {
        /// Distance to shift down.
        delta: u32,
    },
    /// Read from a lane whose index is `self_lane XOR lane_mask`.
    Xor {
        /// XOR mask applied to the lane index.
        lane_mask: u32,
    },
}

impl fmt::Display for ShuffleMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Indexed { src_lane } => write!(f, "Indexed(src={src_lane})"),
            Self::Up { delta } => write!(f, "Up(delta={delta})"),
            Self::Down { delta } => write!(f, "Down(delta={delta})"),
            Self::Xor { lane_mask } => write!(f, "Xor(mask=0x{lane_mask:x})"),
        }
    }
}

/// Operation for warp reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpReduceOp {
    Sum,
    Min,
    Max,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

impl WarpReduceOp {
    /// Identity element for this operation (as f32).
    #[must_use]
    pub fn identity_f32(self) -> f32 {
        match self {
            Self::Sum => 0.0,
            Self::Min => f32::INFINITY,
            Self::Max => f32::NEG_INFINITY,
            // Bitwise ops use u32 identity; callers should use `identity_u32`.
            Self::BitwiseAnd | Self::BitwiseOr | Self::BitwiseXor => 0.0,
        }
    }

    /// Identity element for this operation (as u32).
    #[must_use]
    pub fn identity_u32(self) -> u32 {
        match self {
            Self::Sum => 0,
            Self::Min => u32::MAX,
            Self::Max => 0,
            Self::BitwiseAnd => 0xFFFF_FFFF,
            Self::BitwiseOr => 0,
            Self::BitwiseXor => 0,
        }
    }

    /// Combine two f32 values.
    #[must_use]
    fn combine_f32(self, a: f32, b: f32) -> f32 {
        match self {
            Self::Sum => a + b,
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::BitwiseAnd => f32::from_bits(a.to_bits() & b.to_bits()),
            Self::BitwiseOr => f32::from_bits(a.to_bits() | b.to_bits()),
            Self::BitwiseXor => f32::from_bits(a.to_bits() ^ b.to_bits()),
        }
    }

    /// Combine two u32 values.
    #[must_use]
    fn combine_u32(self, a: u32, b: u32) -> u32 {
        match self {
            Self::Sum => a.wrapping_add(b),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::BitwiseAnd => a & b,
            Self::BitwiseOr => a | b,
            Self::BitwiseXor => a ^ b,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: warp shuffle primitives
// ---------------------------------------------------------------------------

/// Simulate `__shfl_sync` on the CPU: each active lane reads from
/// `src_lane` within its sub-warp segment of width `lane_width`.
///
/// If the computed source lane is inactive, the calling lane keeps its
/// own value (matching CUDA semantics for inactive source lanes).
///
/// # Errors
///
/// Returns an error if `lane_width` is invalid, or `regs.len()` does not
/// match `cfg.warp_size`.
pub fn shfl_sync(
    cfg: &WarpConfig,
    regs: &[f32],
    src_lane: u32,
    lane_width: u32,
) -> Result<Vec<f32>> {
    cfg.validate_lane_width(lane_width)?;
    validate_regs(cfg, regs)?;

    let mut out = regs.to_vec();
    for lane in 0..cfg.warp_size {
        if !cfg.is_active(lane) {
            continue;
        }
        let segment_base = (lane / lane_width) * lane_width;
        let effective_src = segment_base + (src_lane % lane_width);
        if cfg.is_active(effective_src) {
            out[lane as usize] = regs[effective_src as usize];
        }
    }
    Ok(out)
}

/// Simulate `__shfl_up_sync`: each lane reads from `(self - delta)` within
/// its segment. Lanes at the bottom of their segment keep their own value.
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn shfl_up_sync(
    cfg: &WarpConfig,
    regs: &[f32],
    delta: u32,
    lane_width: u32,
) -> Result<Vec<f32>> {
    cfg.validate_lane_width(lane_width)?;
    validate_regs(cfg, regs)?;

    let mut out = regs.to_vec();
    for lane in 0..cfg.warp_size {
        if !cfg.is_active(lane) {
            continue;
        }
        let lane_in_segment = lane % lane_width;
        if lane_in_segment >= delta {
            let src = lane - delta;
            if cfg.is_active(src) {
                out[lane as usize] = regs[src as usize];
            }
        }
    }
    Ok(out)
}

/// Simulate `__shfl_down_sync`: each lane reads from `(self + delta)` within
/// its segment. Lanes at the top of their segment keep their own value.
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn shfl_down_sync(
    cfg: &WarpConfig,
    regs: &[f32],
    delta: u32,
    lane_width: u32,
) -> Result<Vec<f32>> {
    cfg.validate_lane_width(lane_width)?;
    validate_regs(cfg, regs)?;

    let mut out = regs.to_vec();
    for lane in 0..cfg.warp_size {
        if !cfg.is_active(lane) {
            continue;
        }
        let lane_in_segment = lane % lane_width;
        if lane_in_segment + delta < lane_width {
            let src = lane + delta;
            if src < cfg.warp_size && cfg.is_active(src) {
                out[lane as usize] = regs[src as usize];
            }
        }
    }
    Ok(out)
}

/// Simulate `__shfl_xor_sync`: each lane reads from `(self XOR lane_mask)`.
///
/// The XOR is applied within the full warp (not per-segment).
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn shfl_xor_sync(
    cfg: &WarpConfig,
    regs: &[f32],
    lane_mask: u32,
    lane_width: u32,
) -> Result<Vec<f32>> {
    cfg.validate_lane_width(lane_width)?;
    validate_regs(cfg, regs)?;

    let mut out = regs.to_vec();
    for lane in 0..cfg.warp_size {
        if !cfg.is_active(lane) {
            continue;
        }
        let segment_base = (lane / lane_width) * lane_width;
        let lane_in_seg = lane % lane_width;
        let target_in_seg = lane_in_seg ^ lane_mask;
        if target_in_seg < lane_width {
            let src = segment_base + target_in_seg;
            if src < cfg.warp_size && cfg.is_active(src) {
                out[lane as usize] = regs[src as usize];
            }
        }
    }
    Ok(out)
}

/// Dispatch a shuffle operation described by [`ShuffleMode`].
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn shuffle_dispatch(
    cfg: &WarpConfig,
    regs: &[f32],
    mode: &ShuffleMode,
    lane_width: u32,
) -> Result<Vec<f32>> {
    match *mode {
        ShuffleMode::Indexed { src_lane } => shfl_sync(cfg, regs, src_lane, lane_width),
        ShuffleMode::Up { delta } => shfl_up_sync(cfg, regs, delta, lane_width),
        ShuffleMode::Down { delta } => shfl_down_sync(cfg, regs, delta, lane_width),
        ShuffleMode::Xor { lane_mask } => shfl_xor_sync(cfg, regs, lane_mask, lane_width),
    }
}

// ---------------------------------------------------------------------------
// CPU reference: warp broadcast
// ---------------------------------------------------------------------------

/// Broadcast the value from `src_lane` to all active lanes.
///
/// This is equivalent to `shfl_sync` with `lane_width == warp_size` and
/// `src_lane` set to the broadcast source.
///
/// # Errors
///
/// Returns an error if `src_lane >= warp_size` or is not active.
pub fn warp_broadcast(cfg: &WarpConfig, regs: &[f32], src_lane: u32) -> Result<Vec<f32>> {
    validate_regs(cfg, regs)?;
    if src_lane >= cfg.warp_size {
        return Err(WarpShuffleError::InvalidLane { lane: src_lane, warp_size: cfg.warp_size });
    }
    if !cfg.is_active(src_lane) {
        return Err(WarpShuffleError::InvalidLane { lane: src_lane, warp_size: cfg.warp_size });
    }

    let broadcast_val = regs[src_lane as usize];
    let mut out = regs.to_vec();
    for lane in 0..cfg.warp_size {
        if cfg.is_active(lane) {
            out[lane as usize] = broadcast_val;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CPU reference: warp reduction
// ---------------------------------------------------------------------------

/// Reduce all active lanes using `op`, placing the result in every active lane.
///
/// Uses butterfly (XOR) pattern: for each step `i` in `0..log2(warp_size)`,
/// each lane exchanges with `lane XOR (1 << i)` and combines.
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn warp_reduce_f32(cfg: &WarpConfig, regs: &[f32], op: WarpReduceOp) -> Result<Vec<f32>> {
    validate_regs(cfg, regs)?;
    let mut vals = regs.to_vec();

    let steps = log2_u32(cfg.warp_size);
    for i in 0..steps {
        let mask = 1u32 << i;
        let mut next = vals.clone();
        for lane in 0..cfg.warp_size {
            if !cfg.is_active(lane) {
                continue;
            }
            let partner = lane ^ mask;
            if partner < cfg.warp_size && cfg.is_active(partner) {
                next[lane as usize] = op.combine_f32(vals[lane as usize], vals[partner as usize]);
            }
        }
        vals = next;
    }
    Ok(vals)
}

/// Reduce all active lanes over u32 values.
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn warp_reduce_u32(cfg: &WarpConfig, regs: &[u32], op: WarpReduceOp) -> Result<Vec<u32>> {
    if regs.len() != cfg.warp_size as usize {
        return Err(WarpShuffleError::RegisterFileMismatch {
            expected: cfg.warp_size,
            got: regs.len(),
        });
    }
    let mut vals = regs.to_vec();

    let steps = log2_u32(cfg.warp_size);
    for i in 0..steps {
        let mask = 1u32 << i;
        let mut next = vals.clone();
        for lane in 0..cfg.warp_size {
            if !cfg.is_active(lane) {
                continue;
            }
            let partner = lane ^ mask;
            if partner < cfg.warp_size && cfg.is_active(partner) {
                next[lane as usize] = op.combine_u32(vals[lane as usize], vals[partner as usize]);
            }
        }
        vals = next;
    }
    Ok(vals)
}

// ---------------------------------------------------------------------------
// CPU reference: warp scan (prefix sum)
// ---------------------------------------------------------------------------

/// Inclusive prefix sum over active lanes using `shfl_up` pattern.
///
/// After completion, lane `i` holds `sum(regs[0..=i])` for all active
/// lanes in each segment of width `warp_size`.
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn warp_inclusive_scan(cfg: &WarpConfig, regs: &[f32]) -> Result<Vec<f32>> {
    validate_regs(cfg, regs)?;
    let mut vals = regs.to_vec();

    let steps = log2_u32(cfg.warp_size);
    for i in 0..steps {
        let offset = 1u32 << i;
        let mut next = vals.clone();
        for lane in 0..cfg.warp_size {
            if !cfg.is_active(lane) {
                continue;
            }
            if lane >= offset {
                let src = lane - offset;
                if cfg.is_active(src) {
                    next[lane as usize] = vals[lane as usize] + vals[src as usize];
                }
            }
        }
        vals = next;
    }
    Ok(vals)
}

/// Exclusive prefix sum over active lanes.
///
/// After completion, lane `i` holds `sum(regs[0..i])` (lane 0 gets 0.0).
///
/// # Errors
///
/// Returns an error for invalid configuration.
pub fn warp_exclusive_scan(cfg: &WarpConfig, regs: &[f32]) -> Result<Vec<f32>> {
    validate_regs(cfg, regs)?;
    let inclusive = warp_inclusive_scan(cfg, regs)?;
    let mut out = inclusive.clone();
    for lane in 0..cfg.warp_size {
        if !cfg.is_active(lane) {
            continue;
        }
        if lane == 0 {
            out[0] = 0.0;
        } else {
            out[lane as usize] = inclusive[(lane - 1) as usize];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CPU reference: warp vote functions
// ---------------------------------------------------------------------------

/// Simulate `__ballot_sync`: returns a bitmask where bit `i` is set iff
/// lane `i` is active AND `predicates[i]` is `true`.
///
/// # Errors
///
/// Returns an error if `predicates.len() != warp_size`.
pub fn ballot_sync(
    cfg: &WarpConfig,
    predicates: &[bool],
) -> std::result::Result<u32, WarpShuffleError> {
    if predicates.len() != cfg.warp_size as usize {
        return Err(WarpShuffleError::RegisterFileMismatch {
            expected: cfg.warp_size,
            got: predicates.len(),
        });
    }
    let mut ballot = 0u32;
    for lane in 0..cfg.warp_size {
        if cfg.is_active(lane) && predicates[lane as usize] {
            ballot |= 1 << lane;
        }
    }
    Ok(ballot)
}

/// Simulate `__any_sync`: returns `true` if any active lane has a `true`
/// predicate.
///
/// # Errors
///
/// Returns an error if `predicates.len() != warp_size`.
pub fn any_sync(
    cfg: &WarpConfig,
    predicates: &[bool],
) -> std::result::Result<bool, WarpShuffleError> {
    Ok(ballot_sync(cfg, predicates)? != 0)
}

/// Simulate `__all_sync`: returns `true` if every active lane has a `true`
/// predicate.
///
/// # Errors
///
/// Returns an error if `predicates.len() != warp_size`.
pub fn all_sync(
    cfg: &WarpConfig,
    predicates: &[bool],
) -> std::result::Result<bool, WarpShuffleError> {
    let ballot = ballot_sync(cfg, predicates)?;
    Ok(ballot == cfg.active_mask)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_regs(cfg: &WarpConfig, regs: &[f32]) -> Result<()> {
    if regs.len() != cfg.warp_size as usize {
        return Err(WarpShuffleError::RegisterFileMismatch {
            expected: cfg.warp_size,
            got: regs.len(),
        });
    }
    Ok(())
}

/// Integer log2 for powers of two (panics on 0).
#[must_use]
fn log2_u32(n: u32) -> u32 {
    debug_assert!(n > 0);
    31 - n.leading_zeros()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn cfg32() -> WarpConfig {
        WarpConfig::default_32()
    }

    fn ascending_f32(n: u32) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    fn ones_f32(n: u32) -> Vec<f32> {
        vec![1.0; n as usize]
    }

    fn constant_f32(n: u32, val: f32) -> Vec<f32> {
        vec![val; n as usize]
    }

    // ====================================================================
    // WarpConfig tests
    // ====================================================================

    #[test]
    fn config_new_valid_sizes() {
        for &sz in &[1, 2, 4, 8, 16, 32] {
            let cfg = WarpConfig::new(sz).unwrap();
            assert_eq!(cfg.warp_size, sz);
            if sz == 32 {
                assert_eq!(cfg.active_mask, 0xFFFF_FFFF);
            } else {
                assert_eq!(cfg.active_mask, (1u32 << sz) - 1);
            }
        }
    }

    #[test]
    fn config_rejects_invalid_sizes() {
        for &sz in &[0, 3, 5, 6, 7, 9, 33, 64] {
            assert!(WarpConfig::new(sz).is_err());
        }
    }

    #[test]
    fn config_with_mask_valid() {
        let cfg = WarpConfig::with_mask(32, 0b1010_1010).unwrap();
        assert!(cfg.is_active(1));
        assert!(!cfg.is_active(0));
        assert_eq!(cfg.active_count(), 4); // bits 1,3,5,7
    }

    #[test]
    fn config_with_mask_rejects_oob() {
        assert!(WarpConfig::with_mask(4, 0b1_0000).is_err());
    }

    #[test]
    fn config_default_32() {
        let cfg = WarpConfig::default_32();
        assert_eq!(cfg.warp_size, 32);
        assert_eq!(cfg.active_mask, 0xFFFF_FFFF);
        assert_eq!(cfg.active_count(), 32);
    }

    // ====================================================================
    // shfl_sync (indexed) tests
    // ====================================================================

    #[test]
    fn shfl_sync_broadcast_lane0() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_sync(&cfg, &regs, 0, 32).unwrap();
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn shfl_sync_broadcast_lane15() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_sync(&cfg, &regs, 15, 32).unwrap();
        for &v in &out {
            assert_eq!(v, 15.0);
        }
    }

    #[test]
    fn shfl_sync_segments_width_8() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        // Within each 8-lane segment, read src_lane=3
        let out = shfl_sync(&cfg, &regs, 3, 8).unwrap();
        // Segment 0 (lanes 0-7): reads lane 3 → 3.0
        // Segment 1 (lanes 8-15): reads lane 11 → 11.0
        // Segment 2 (lanes 16-23): reads lane 19 → 19.0
        // Segment 3 (lanes 24-31): reads lane 27 → 27.0
        for lane in 0..32u32 {
            let segment_base = (lane / 8) * 8;
            assert_eq!(out[lane as usize], (segment_base + 3) as f32);
        }
    }

    #[test]
    fn shfl_sync_width_1_identity() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        // Width 1: each lane is its own segment, reads src_lane=0 → itself
        let out = shfl_sync(&cfg, &regs, 0, 1).unwrap();
        assert_eq!(out, regs);
    }

    #[test]
    fn shfl_sync_register_mismatch() {
        let cfg = cfg32();
        let regs = vec![1.0; 16];
        assert!(shfl_sync(&cfg, &regs, 0, 32).is_err());
    }

    #[test]
    fn shfl_sync_invalid_lane_width() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        assert!(shfl_sync(&cfg, &regs, 0, 3).is_err()); // not power of two
        assert!(shfl_sync(&cfg, &regs, 0, 64).is_err()); // exceeds warp
    }

    // ====================================================================
    // shfl_up_sync tests
    // ====================================================================

    #[test]
    fn shfl_up_delta1() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_up_sync(&cfg, &regs, 1, 32).unwrap();
        assert_eq!(out[0], 0.0); // lane 0 keeps its value
        for i in 1..32usize {
            assert_eq!(out[i], (i - 1) as f32);
        }
    }

    #[test]
    fn shfl_up_delta0_identity() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_up_sync(&cfg, &regs, 0, 32).unwrap();
        assert_eq!(out, regs);
    }

    #[test]
    fn shfl_up_delta_exceeds_segment() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![10.0, 20.0, 30.0, 40.0];
        let out = shfl_up_sync(&cfg, &regs, 4, 4).unwrap();
        // All lanes are at the bottom of their segment (delta=4 >= segment pos)
        assert_eq!(out, regs);
    }

    #[test]
    fn shfl_up_segments_width_4() {
        let cfg = WarpConfig::new(8).unwrap();
        let regs: Vec<f32> = (0..8).map(|i| (i * 10) as f32).collect();
        let out = shfl_up_sync(&cfg, &regs, 1, 4).unwrap();
        // Segment 0: lanes 0-3 → [0, 0, 10, 20]
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 10.0);
        assert_eq!(out[3], 20.0);
        // Segment 1: lanes 4-7 → [40, 40, 50, 60]
        assert_eq!(out[4], 40.0);
        assert_eq!(out[5], 40.0);
        assert_eq!(out[6], 50.0);
        assert_eq!(out[7], 60.0);
    }

    // ====================================================================
    // shfl_down_sync tests
    // ====================================================================

    #[test]
    fn shfl_down_delta1() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_down_sync(&cfg, &regs, 1, 32).unwrap();
        for i in 0..31usize {
            assert_eq!(out[i], (i + 1) as f32);
        }
        assert_eq!(out[31], 31.0); // top lane keeps its value
    }

    #[test]
    fn shfl_down_delta0_identity() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_down_sync(&cfg, &regs, 0, 32).unwrap();
        assert_eq!(out, regs);
    }

    #[test]
    fn shfl_down_small_warp() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![10.0, 20.0, 30.0, 40.0];
        let out = shfl_down_sync(&cfg, &regs, 2, 4).unwrap();
        assert_eq!(out[0], 30.0);
        assert_eq!(out[1], 40.0);
        assert_eq!(out[2], 30.0); // keeps own value (at top of segment)
        assert_eq!(out[3], 40.0); // keeps own value
    }

    // ====================================================================
    // shfl_xor_sync tests
    // ====================================================================

    #[test]
    fn shfl_xor_mask1_swaps_pairs() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_xor_sync(&cfg, &regs, 1, 32).unwrap();
        for i in (0..32).step_by(2) {
            assert_eq!(out[i], (i + 1) as f32);
            assert_eq!(out[i + 1], i as f32);
        }
    }

    #[test]
    fn shfl_xor_mask0_identity() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = shfl_xor_sync(&cfg, &regs, 0, 32).unwrap();
        assert_eq!(out, regs);
    }

    #[test]
    fn shfl_xor_width_4_mask2() {
        let cfg = WarpConfig::new(8).unwrap();
        let regs: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = shfl_xor_sync(&cfg, &regs, 2, 4).unwrap();
        // Within each 4-lane segment, XOR with 2:
        // Seg 0: 0^2=2, 1^2=3, 2^2=0, 3^2=1 → [2,3,0,1]
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 3.0);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 1.0);
        // Seg 1: 0^2=2→6, 1^2=3→7, 2^2=0→4, 3^2=1→5 → [6,7,4,5]
        assert_eq!(out[4], 6.0);
        assert_eq!(out[5], 7.0);
        assert_eq!(out[6], 4.0);
        assert_eq!(out[7], 5.0);
    }

    #[test]
    fn shfl_xor_mask_exceeds_segment_keeps_value() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![10.0, 20.0, 30.0, 40.0];
        // mask=4 → target_in_seg = lane_in_seg ^ 4 >= lane_width(4) → no exchange
        let out = shfl_xor_sync(&cfg, &regs, 4, 4).unwrap();
        assert_eq!(out, regs);
    }

    // ====================================================================
    // shuffle_dispatch tests
    // ====================================================================

    #[test]
    fn dispatch_indexed() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let mode = ShuffleMode::Indexed { src_lane: 5 };
        let out = shuffle_dispatch(&cfg, &regs, &mode, 32).unwrap();
        for &v in &out {
            assert_eq!(v, 5.0);
        }
    }

    #[test]
    fn dispatch_up() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let mode = ShuffleMode::Up { delta: 1 };
        let out = shuffle_dispatch(&cfg, &regs, &mode, 32).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn dispatch_down() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let mode = ShuffleMode::Down { delta: 1 };
        let out = shuffle_dispatch(&cfg, &regs, &mode, 32).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[31], 31.0);
    }

    #[test]
    fn dispatch_xor() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let mode = ShuffleMode::Xor { lane_mask: 1 };
        let out = shuffle_dispatch(&cfg, &regs, &mode, 32).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 0.0);
    }

    // ====================================================================
    // warp_broadcast tests
    // ====================================================================

    #[test]
    fn broadcast_from_lane0() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = warp_broadcast(&cfg, &regs, 0).unwrap();
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn broadcast_from_lane31() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = warp_broadcast(&cfg, &regs, 31).unwrap();
        for &v in &out {
            assert_eq!(v, 31.0);
        }
    }

    #[test]
    fn broadcast_from_middle() {
        let cfg = WarpConfig::new(8).unwrap();
        let regs = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let out = warp_broadcast(&cfg, &regs, 4).unwrap();
        for &v in &out {
            assert_eq!(v, 50.0);
        }
    }

    #[test]
    fn broadcast_invalid_lane() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![1.0; 4];
        assert!(warp_broadcast(&cfg, &regs, 4).is_err());
    }

    #[test]
    fn broadcast_inactive_lane_error() {
        let cfg = WarpConfig::with_mask(4, 0b1010).unwrap(); // lanes 1,3 active
        let regs = vec![1.0, 2.0, 3.0, 4.0];
        assert!(warp_broadcast(&cfg, &regs, 0).is_err()); // lane 0 inactive
        let out = warp_broadcast(&cfg, &regs, 1).unwrap();
        assert_eq!(out[1], 2.0);
        assert_eq!(out[3], 2.0);
    }

    // ====================================================================
    // warp_reduce_f32 tests
    // ====================================================================

    #[test]
    fn reduce_sum_ascending() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let expected: f32 = (0..32).map(|i| i as f32).sum();
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
        for &v in &out {
            assert!((v - expected).abs() < 1e-3, "expected {expected}, got {v}");
        }
    }

    #[test]
    fn reduce_sum_ones() {
        let cfg = cfg32();
        let regs = ones_f32(32);
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
        for &v in &out {
            assert!((v - 32.0).abs() < 1e-6);
        }
    }

    #[test]
    fn reduce_max() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Max).unwrap();
        for &v in &out {
            assert_eq!(v, 31.0);
        }
    }

    #[test]
    fn reduce_min() {
        let cfg = cfg32();
        let regs = ascending_f32(32);
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Min).unwrap();
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn reduce_sum_small_warp() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![1.0, 2.0, 3.0, 4.0];
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
        for &v in &out {
            assert!((v - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn reduce_max_all_same() {
        let cfg = cfg32();
        let regs = constant_f32(32, 42.0);
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Max).unwrap();
        for &v in &out {
            assert_eq!(v, 42.0);
        }
    }

    #[test]
    fn reduce_min_all_same() {
        let cfg = cfg32();
        let regs = constant_f32(32, -7.0);
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Min).unwrap();
        for &v in &out {
            assert_eq!(v, -7.0);
        }
    }

    #[test]
    fn reduce_single_lane() {
        let cfg = WarpConfig::new(1).unwrap();
        let regs = vec![99.0];
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
        assert_eq!(out, vec![99.0]);
    }

    // ====================================================================
    // warp_reduce_u32 tests
    // ====================================================================

    #[test]
    fn reduce_u32_bitwise_and() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![0b1111u32, 0b1010, 0b1100, 0b1000];
        let out = warp_reduce_u32(&cfg, &regs, WarpReduceOp::BitwiseAnd).unwrap();
        for &v in &out {
            assert_eq!(v, 0b1000);
        }
    }

    #[test]
    fn reduce_u32_bitwise_or() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![0b0001u32, 0b0010, 0b0100, 0b1000];
        let out = warp_reduce_u32(&cfg, &regs, WarpReduceOp::BitwiseOr).unwrap();
        for &v in &out {
            assert_eq!(v, 0b1111);
        }
    }

    #[test]
    fn reduce_u32_bitwise_xor() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![0b1111u32, 0b1111, 0b1111, 0b1111];
        let out = warp_reduce_u32(&cfg, &regs, WarpReduceOp::BitwiseXor).unwrap();
        // XOR of 4 identical values: 0b1111 ^ 0b1111 = 0, 0 ^ 0b1111 = 0b1111, etc.
        // butterfly: step0 pairs (0,1)(2,3): each becomes 0
        // step1 pairs (0,2)(1,3): each becomes 0^0 = 0
        for &v in &out {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn reduce_u32_sum() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![1u32, 2, 3, 4];
        let out = warp_reduce_u32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
        for &v in &out {
            assert_eq!(v, 10);
        }
    }

    // ====================================================================
    // warp scan tests
    // ====================================================================

    #[test]
    fn inclusive_scan_ascending() {
        let cfg = WarpConfig::new(8).unwrap();
        let regs = vec![1.0; 8];
        let out = warp_inclusive_scan(&cfg, &regs).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - (i + 1) as f32).abs() < 1e-6, "lane {i}: got {v}");
        }
    }

    #[test]
    fn inclusive_scan_powers_of_two() {
        let cfg = WarpConfig::new(4).unwrap();
        let regs = vec![1.0, 2.0, 4.0, 8.0];
        let out = warp_inclusive_scan(&cfg, &regs).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        assert!((out[2] - 7.0).abs() < 1e-6);
        assert!((out[3] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn exclusive_scan_ascending() {
        let cfg = WarpConfig::new(8).unwrap();
        let regs = vec![1.0; 8];
        let out = warp_exclusive_scan(&cfg, &regs).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-6, "lane {i}: got {v}");
        }
    }

    #[test]
    fn exclusive_scan_single() {
        let cfg = WarpConfig::new(1).unwrap();
        let regs = vec![42.0];
        let out = warp_exclusive_scan(&cfg, &regs).unwrap();
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn inclusive_scan_32_lanes() {
        let cfg = cfg32();
        let regs = ones_f32(32);
        let out = warp_inclusive_scan(&cfg, &regs).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - (i + 1) as f32).abs() < 1e-3);
        }
    }

    #[test]
    fn exclusive_scan_32_lanes() {
        let cfg = cfg32();
        let regs = ones_f32(32);
        let out = warp_exclusive_scan(&cfg, &regs).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - i as f32).abs() < 1e-3);
        }
    }

    // ====================================================================
    // vote function tests
    // ====================================================================

    #[test]
    fn ballot_all_true() {
        let cfg = cfg32();
        let preds = vec![true; 32];
        assert_eq!(ballot_sync(&cfg, &preds).unwrap(), 0xFFFF_FFFF);
    }

    #[test]
    fn ballot_all_false() {
        let cfg = cfg32();
        let preds = vec![false; 32];
        assert_eq!(ballot_sync(&cfg, &preds).unwrap(), 0);
    }

    #[test]
    fn ballot_alternating() {
        let cfg = cfg32();
        let preds: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        let expected = 0x5555_5555u32; // even bits set
        assert_eq!(ballot_sync(&cfg, &preds).unwrap(), expected);
    }

    #[test]
    fn ballot_respects_mask() {
        let cfg = WarpConfig::with_mask(4, 0b1010).unwrap();
        let preds = vec![true, true, true, true];
        // Only lanes 1 and 3 are active
        assert_eq!(ballot_sync(&cfg, &preds).unwrap(), 0b1010);
    }

    #[test]
    fn ballot_wrong_length() {
        let cfg = cfg32();
        let preds = vec![true; 16];
        assert!(ballot_sync(&cfg, &preds).is_err());
    }

    #[test]
    fn any_sync_some_true() {
        let cfg = cfg32();
        let mut preds = vec![false; 32];
        preds[15] = true;
        assert!(any_sync(&cfg, &preds).unwrap());
    }

    #[test]
    fn any_sync_none_true() {
        let cfg = cfg32();
        let preds = vec![false; 32];
        assert!(!any_sync(&cfg, &preds).unwrap());
    }

    #[test]
    fn all_sync_all_true() {
        let cfg = cfg32();
        let preds = vec![true; 32];
        assert!(all_sync(&cfg, &preds).unwrap());
    }

    #[test]
    fn all_sync_one_false() {
        let cfg = cfg32();
        let mut preds = vec![true; 32];
        preds[0] = false;
        assert!(!all_sync(&cfg, &preds).unwrap());
    }

    #[test]
    fn all_sync_with_partial_mask() {
        let cfg = WarpConfig::with_mask(4, 0b0110).unwrap(); // lanes 1,2 active
        let preds = vec![false, true, true, false];
        // Only active lanes matter; both active lanes are true
        assert!(all_sync(&cfg, &preds).unwrap());
    }

    // ====================================================================
    // Inactive lane tests
    // ====================================================================

    #[test]
    fn shfl_down_inactive_preserves_value() {
        let cfg = WarpConfig::with_mask(4, 0b0101).unwrap(); // lanes 0,2 active
        let regs = vec![10.0, 20.0, 30.0, 40.0];
        let out = shfl_down_sync(&cfg, &regs, 1, 4).unwrap();
        // Lane 0: active, wants src=1, but lane 1 is inactive → keeps 10.0
        assert_eq!(out[0], 10.0);
        // Lane 1: inactive → unchanged
        assert_eq!(out[1], 20.0);
        // Lane 2: active, wants src=3, lane 3 inactive → keeps 30.0
        assert_eq!(out[2], 30.0);
        // Lane 3: inactive → unchanged
        assert_eq!(out[3], 40.0);
    }

    #[test]
    fn reduce_with_partial_mask() {
        let cfg = WarpConfig::with_mask(4, 0b1010).unwrap(); // lanes 1,3 active
        let regs = vec![10.0, 20.0, 30.0, 40.0];
        let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
        // Active lanes 1 and 3 exchange via butterfly.
        // Step 0 (mask=1): lane 1 partners with lane 0 (inactive → no change),
        //   lane 3 partners with lane 2 (inactive → no change).
        // Step 1 (mask=2): lane 1 partners with lane 3 → 20+40 = 60.
        assert!((out[1] - 60.0).abs() < 1e-6);
        assert!((out[3] - 60.0).abs() < 1e-6);
    }

    // ====================================================================
    // Error display tests
    // ====================================================================

    #[test]
    fn error_display_messages() {
        let e = WarpShuffleError::InvalidLane { lane: 33, warp_size: 32 };
        assert!(e.to_string().contains("33"));
        assert!(e.to_string().contains("32"));

        let e = WarpShuffleError::InvalidWarpSize(7);
        assert!(e.to_string().contains("7"));

        let e = WarpShuffleError::InvalidMask { mask: 0xFF, warp_size: 4 };
        assert!(e.to_string().contains("4"));
    }

    #[test]
    fn shuffle_mode_display() {
        assert_eq!(ShuffleMode::Indexed { src_lane: 5 }.to_string(), "Indexed(src=5)");
        assert_eq!(ShuffleMode::Up { delta: 2 }.to_string(), "Up(delta=2)");
        assert_eq!(ShuffleMode::Down { delta: 3 }.to_string(), "Down(delta=3)");
        assert_eq!(ShuffleMode::Xor { lane_mask: 0xF }.to_string(), "Xor(mask=0xf)");
    }

    // ====================================================================
    // Property-based tests
    // ====================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_warp_size() -> impl Strategy<Value = u32> {
            prop_oneof![Just(1u32), Just(2), Just(4), Just(8), Just(16), Just(32)]
        }

        fn arb_regs(size: u32) -> impl Strategy<Value = Vec<f32>> {
            proptest::collection::vec(-1000.0f32..1000.0, size as usize)
        }

        proptest! {
            #[test]
            fn prop_shfl_xor_involution(
                ws in arb_warp_size(),
                mask in 0u32..32,
            ) {
                let cfg = WarpConfig::new(ws).unwrap();
                let regs: Vec<f32> = (0..ws).map(|i| i as f32).collect();
                if let Ok(once) = shfl_xor_sync(&cfg, &regs, mask, ws) {
                    if let Ok(twice) = shfl_xor_sync(&cfg, &once, mask, ws) {
                        // XOR is its own inverse: applying twice gives original
                        for i in 0..ws as usize {
                            prop_assert!((twice[i] - regs[i]).abs() < 1e-6);
                        }
                    }
                }
            }

            #[test]
            fn prop_inclusive_scan_monotone(ws in arb_warp_size()) {
                let cfg = WarpConfig::new(ws).unwrap();
                let regs = vec![1.0f32; ws as usize];
                let out = warp_inclusive_scan(&cfg, &regs).unwrap();
                for i in 1..ws as usize {
                    prop_assert!(out[i] >= out[i - 1]);
                }
            }

            #[test]
            fn prop_exclusive_scan_starts_zero(ws in arb_warp_size()) {
                let cfg = WarpConfig::new(ws).unwrap();
                let regs: Vec<f32> = (0..ws).map(|i| (i + 1) as f32).collect();
                let out = warp_exclusive_scan(&cfg, &regs).unwrap();
                prop_assert_eq!(out[0], 0.0);
            }

            #[test]
            fn prop_broadcast_uniform(
                ws in arb_warp_size(),
            ) {
                let cfg = WarpConfig::new(ws).unwrap();
                let regs: Vec<f32> = (0..ws).map(|i| i as f32).collect();
                let lane = 0u32;
                let out = warp_broadcast(&cfg, &regs, lane).unwrap();
                for &v in &out {
                    prop_assert_eq!(v, regs[lane as usize]);
                }
            }

            #[test]
            fn prop_reduce_sum_matches_sequential(ws in arb_warp_size()) {
                let cfg = WarpConfig::new(ws).unwrap();
                let regs: Vec<f32> = (0..ws).map(|i| (i + 1) as f32).collect();
                let expected: f32 = regs.iter().sum();
                let out = warp_reduce_f32(&cfg, &regs, WarpReduceOp::Sum).unwrap();
                for &v in &out {
                    prop_assert!((v - expected).abs() < 1e-2,
                        "ws={ws}, expected={expected}, got={v}");
                }
            }
        }
    }
}
