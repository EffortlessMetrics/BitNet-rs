//! CUDA warp-level operations with CPU reference implementations.
//!
//! This module provides high-level abstractions for CUDA warp primitives
//! (`__shfl_sync`, `__shfl_xor_sync`, `__ballot_sync`, `__all_sync`,
//! `__any_sync`) with CPU fallback implementations that simulate warp
//! behavior for correctness testing and non-GPU environments.
//!
//! # Types
//!
//! - [`WarpConfig`] — warp geometry and active lane mask
//! - [`WarpShuffle`] — shuffle operation descriptors (direct, XOR, butterfly)
//! - [`WarpReduce`] — reduction operation descriptors (sum, max, min, product)
//! - [`WarpScan`] — scan/prefix-sum operation descriptors
//! - [`WarpError`] — domain-specific error type for warp operations
//!
//! # Functions
//!
//! - [`warp_reduce_sum`] / [`warp_reduce_max`] — reductions across lanes
//! - [`warp_broadcast`] — broadcast from one lane to all
//! - [`warp_prefix_sum`] — inclusive prefix sum
//! - [`warp_ballot`] — predicate vote → bitmask
//! - [`warp_all`] / [`warp_any`] — unanimous / existential predicates

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Domain-specific error type for warp operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarpError {
    /// Data buffer length does not match the warp size.
    LengthMismatch { expected: u32, actual: usize },
    /// The specified lane is outside the valid warp range.
    LaneOutOfRange { lane: u32, warp_size: u32 },
    /// The specified source lane is not active in the mask.
    InactiveLane { lane: u32, active_mask: u32 },
    /// An empty mask was provided (no active lanes).
    EmptyMask,
    /// A predicate buffer is too short.
    PredicateLengthMismatch { expected: u32, actual: usize },
    /// Invalid shuffle descriptor.
    InvalidShuffle { reason: String },
}

impl fmt::Display for WarpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(f, "warp data length {actual} != warp_size {expected}")
            }
            Self::LaneOutOfRange { lane, warp_size } => {
                write!(f, "lane {lane} out of range for warp_size {warp_size}")
            }
            Self::InactiveLane { lane, active_mask } => {
                write!(f, "lane {lane} is not active (mask={active_mask:#010x})")
            }
            Self::EmptyMask => write!(f, "active mask must have at least one lane"),
            Self::PredicateLengthMismatch { expected, actual } => {
                write!(f, "predicates length {actual} < warp_size {expected}")
            }
            Self::InvalidShuffle { reason } => {
                write!(f, "invalid shuffle: {reason}")
            }
        }
    }
}

impl std::error::Error for WarpError {}

/// Result alias for warp operations.
pub type WarpResult<T> = std::result::Result<T, WarpError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default CUDA warp size.
pub const WARP_SIZE: u32 = 32;

/// Configuration for warp-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpConfig {
    /// Number of lanes in the warp (32 for CUDA).
    pub warp_size: u32,
    /// Active lane mask. Bit `i` set means lane `i` participates.
    pub active_mask: u32,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self { warp_size: WARP_SIZE, active_mask: 0xFFFF_FFFF }
    }
}

impl WarpConfig {
    /// Create a config with all 32 lanes active.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config with a custom active lane mask.
    pub fn with_mask(active_mask: u32) -> WarpResult<Self> {
        if active_mask == 0 {
            return Err(WarpError::EmptyMask);
        }
        Ok(Self { warp_size: WARP_SIZE, active_mask })
    }

    /// Number of active lanes.
    pub fn active_count(&self) -> u32 {
        self.active_mask.count_ones()
    }

    /// Check whether lane `i` is active.
    pub fn is_active(&self, lane: u32) -> bool {
        lane < self.warp_size && (self.active_mask & (1 << lane)) != 0
    }

    /// Return the index of the first active lane.
    pub fn first_active_lane(&self) -> Option<u32> {
        (0..self.warp_size).find(|&i| self.is_active(i))
    }
}

// ---------------------------------------------------------------------------
// Shuffle descriptor
// ---------------------------------------------------------------------------

/// Describes a warp shuffle operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarpShuffle {
    /// Direct indexed shuffle — each lane reads from `src_lanes[lane]`.
    Direct { src_lanes: Vec<u32> },
    /// XOR shuffle — each lane reads from `lane ^ xor_mask`.
    Xor { xor_mask: u32 },
    /// Butterfly (down) shuffle with a given delta.
    Down { delta: u32 },
    /// Butterfly (up) shuffle with a given delta.
    Up { delta: u32 },
}

// ---------------------------------------------------------------------------
// Reduce descriptor
// ---------------------------------------------------------------------------

/// Describes a warp reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpReduce {
    /// Sum of all active lane values.
    Sum,
    /// Maximum of all active lane values.
    Max,
    /// Minimum of all active lane values.
    Min,
    /// Product of all active lane values.
    Product,
}

// ---------------------------------------------------------------------------
// Scan descriptor
// ---------------------------------------------------------------------------

/// Describes a warp scan (prefix) operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpScan {
    /// Inclusive prefix sum.
    InclusiveSum,
    /// Exclusive prefix sum (first element = 0).
    ExclusiveSum,
    /// Inclusive prefix max.
    InclusiveMax,
}

// ---------------------------------------------------------------------------
// CUDA kernel source — feature-gated
// ---------------------------------------------------------------------------

/// CUDA C source for warp-level intrinsic kernels.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const CUDA_WARP_OPS_SRC: &str = r#"
extern "C" __global__ void warp_reduce_sum_f32(
    float* __restrict__ data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1)
        val += __shfl_xor_sync(MASK, val, offset);
    data[idx] = val;
}

extern "C" __global__ void warp_reduce_max_f32(
    float* __restrict__ data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(MASK, val, offset));
    data[idx] = val;
}

extern "C" __global__ void warp_broadcast_f32(
    float* __restrict__ data, int n, int src_lane)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    val = __shfl_sync(0xFFFFFFFFu, val, src_lane);
    data[idx] = val;
}

extern "C" __global__ void warp_ballot_u32(
    const int* __restrict__ predicates,
    unsigned int* __restrict__ result,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int ballot = __ballot_sync(0xFFFFFFFFu, predicates[idx]);
    if ((threadIdx.x & 31) == 0)
        result[idx / 32] = ballot;
}
"#;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_data(data: &[f32], config: &WarpConfig) -> WarpResult<()> {
    if data.len() < config.warp_size as usize {
        return Err(WarpError::LengthMismatch { expected: config.warp_size, actual: data.len() });
    }
    Ok(())
}

fn validate_predicates(preds: &[bool], config: &WarpConfig) -> WarpResult<()> {
    if preds.len() < config.warp_size as usize {
        return Err(WarpError::PredicateLengthMismatch {
            expected: config.warp_size,
            actual: preds.len(),
        });
    }
    Ok(())
}

fn validate_lane(lane: u32, config: &WarpConfig) -> WarpResult<()> {
    if lane >= config.warp_size {
        return Err(WarpError::LaneOutOfRange { lane, warp_size: config.warp_size });
    }
    if !config.is_active(lane) {
        return Err(WarpError::InactiveLane { lane, active_mask: config.active_mask });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: reductions
// ---------------------------------------------------------------------------

/// Butterfly-pattern sum reduction across active warp lanes.
///
/// All active lanes receive the sum. Inactive lanes are unchanged.
pub fn warp_reduce_sum(data: &mut [f32], config: &WarpConfig) -> WarpResult<()> {
    validate_data(data, config)?;
    let sum: f32 =
        (0..config.warp_size).filter(|&i| config.is_active(i)).map(|i| data[i as usize]).sum();
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = sum;
        }
    }
    Ok(())
}

/// Butterfly-pattern max reduction across active warp lanes.
///
/// All active lanes receive the maximum value. Inactive lanes unchanged.
pub fn warp_reduce_max(data: &mut [f32], config: &WarpConfig) -> WarpResult<()> {
    validate_data(data, config)?;
    let max_val = (0..config.warp_size)
        .filter(|&i| config.is_active(i))
        .map(|i| data[i as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = max_val;
        }
    }
    Ok(())
}

/// Generic warp reduction using [`WarpReduce`] descriptor.
pub fn warp_reduce(data: &mut [f32], op: WarpReduce, config: &WarpConfig) -> WarpResult<()> {
    validate_data(data, config)?;
    let active_vals: Vec<f32> =
        (0..config.warp_size).filter(|&i| config.is_active(i)).map(|i| data[i as usize]).collect();
    if active_vals.is_empty() {
        return Ok(());
    }
    let result = match op {
        WarpReduce::Sum => active_vals.iter().sum(),
        WarpReduce::Max => active_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        WarpReduce::Min => active_vals.iter().copied().fold(f32::INFINITY, f32::min),
        WarpReduce::Product => active_vals.iter().product(),
    };
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = result;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: broadcast
// ---------------------------------------------------------------------------

/// Broadcast the value in `src_lane` to all active lanes.
///
/// Simulates `__shfl_sync(mask, val, src_lane)`.
pub fn warp_broadcast(data: &mut [f32], src_lane: u32, config: &WarpConfig) -> WarpResult<()> {
    validate_data(data, config)?;
    validate_lane(src_lane, config)?;
    let val = data[src_lane as usize];
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = val;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: shuffle
// ---------------------------------------------------------------------------

/// Execute a shuffle described by [`WarpShuffle`].
pub fn warp_shuffle(
    data: &mut [f32],
    shuffle: &WarpShuffle,
    config: &WarpConfig,
) -> WarpResult<()> {
    validate_data(data, config)?;
    let ws = config.warp_size as usize;
    let snapshot: Vec<f32> = data[..ws].to_vec();
    match shuffle {
        WarpShuffle::Direct { src_lanes } => {
            if src_lanes.len() < ws {
                return Err(WarpError::InvalidShuffle {
                    reason: format!(
                        "src_lanes length {} < warp_size {}",
                        src_lanes.len(),
                        config.warp_size
                    ),
                });
            }
            for i in 0..config.warp_size {
                if config.is_active(i) {
                    let src = src_lanes[i as usize];
                    if !config.is_active(src) {
                        return Err(WarpError::InactiveLane {
                            lane: src,
                            active_mask: config.active_mask,
                        });
                    }
                    data[i as usize] = snapshot[src as usize];
                }
            }
        }
        WarpShuffle::Xor { xor_mask } => {
            for i in 0..config.warp_size {
                if config.is_active(i) {
                    let src = i ^ xor_mask;
                    if src < config.warp_size {
                        data[i as usize] = snapshot[src as usize];
                    }
                }
            }
        }
        WarpShuffle::Down { delta } => {
            for i in 0..config.warp_size {
                if config.is_active(i) {
                    let src = i + delta;
                    if src < config.warp_size {
                        data[i as usize] = snapshot[src as usize];
                    }
                }
            }
        }
        WarpShuffle::Up { delta } => {
            for i in 0..config.warp_size {
                if config.is_active(i) && i >= *delta {
                    data[i as usize] = snapshot[(i - delta) as usize];
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: prefix sum / scan
// ---------------------------------------------------------------------------

/// Inclusive prefix sum across active lanes.
///
/// After execution, `data[i]` contains the sum of all active values
/// in lanes `0..=i`.
pub fn warp_prefix_sum(data: &mut [f32], config: &WarpConfig) -> WarpResult<()> {
    validate_data(data, config)?;
    let mut running = 0.0f32;
    for i in 0..config.warp_size {
        if config.is_active(i) {
            running += data[i as usize];
            data[i as usize] = running;
        }
    }
    Ok(())
}

/// Generic warp scan using [`WarpScan`] descriptor.
pub fn warp_scan(data: &mut [f32], op: WarpScan, config: &WarpConfig) -> WarpResult<()> {
    validate_data(data, config)?;
    match op {
        WarpScan::InclusiveSum => {
            let mut running = 0.0f32;
            for i in 0..config.warp_size {
                if config.is_active(i) {
                    running += data[i as usize];
                    data[i as usize] = running;
                }
            }
        }
        WarpScan::ExclusiveSum => {
            let mut running = 0.0f32;
            for i in 0..config.warp_size {
                if config.is_active(i) {
                    let val = data[i as usize];
                    data[i as usize] = running;
                    running += val;
                }
            }
        }
        WarpScan::InclusiveMax => {
            let mut running = f32::NEG_INFINITY;
            for i in 0..config.warp_size {
                if config.is_active(i) {
                    running = running.max(data[i as usize]);
                    data[i as usize] = running;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CPU reference: voting
// ---------------------------------------------------------------------------

/// Ballot vote across warp lanes.
///
/// Returns a bitmask where bit `i` is set iff lane `i` is active and
/// `predicates[i]` is true. Simulates `__ballot_sync`.
pub fn warp_ballot(predicates: &[bool], config: &WarpConfig) -> WarpResult<u32> {
    validate_predicates(predicates, config)?;
    let mut ballot = 0u32;
    for i in 0..config.warp_size {
        if config.is_active(i) && predicates[i as usize] {
            ballot |= 1 << i;
        }
    }
    Ok(ballot)
}

/// Check if all active lanes satisfy the predicate.
///
/// Simulates `__all_sync(mask, predicate)`.
pub fn warp_all(predicates: &[bool], config: &WarpConfig) -> WarpResult<bool> {
    validate_predicates(predicates, config)?;
    Ok((0..config.warp_size).filter(|&i| config.is_active(i)).all(|i| predicates[i as usize]))
}

/// Check if any active lane satisfies the predicate.
///
/// Simulates `__any_sync(mask, predicate)`.
pub fn warp_any(predicates: &[bool], config: &WarpConfig) -> WarpResult<bool> {
    validate_predicates(predicates, config)?;
    Ok((0..config.warp_size).filter(|&i| config.is_active(i)).any(|i| predicates[i as usize]))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helper: full-warp data
    fn full_config() -> WarpConfig {
        WarpConfig::new()
    }

    fn iota_data() -> Vec<f32> {
        (0..32).map(|i| (i + 1) as f32).collect()
    }

    // -------------------------------------------------------------------
    // WarpError display
    // -------------------------------------------------------------------

    #[test]
    fn error_display_length_mismatch() {
        let e = WarpError::LengthMismatch { expected: 32, actual: 8 };
        assert!(e.to_string().contains("8"));
    }

    #[test]
    fn error_display_lane_out_of_range() {
        let e = WarpError::LaneOutOfRange { lane: 33, warp_size: 32 };
        assert!(e.to_string().contains("33"));
    }

    #[test]
    fn error_display_inactive_lane() {
        let e = WarpError::InactiveLane { lane: 5, active_mask: 0x0F };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn error_display_empty_mask() {
        assert!(WarpError::EmptyMask.to_string().contains("at least one"));
    }

    #[test]
    fn error_display_predicate_mismatch() {
        let e = WarpError::PredicateLengthMismatch { expected: 32, actual: 4 };
        assert!(e.to_string().contains("4"));
    }

    #[test]
    fn error_display_invalid_shuffle() {
        let e = WarpError::InvalidShuffle { reason: "bad".into() };
        assert!(e.to_string().contains("bad"));
    }

    // -------------------------------------------------------------------
    // WarpConfig
    // -------------------------------------------------------------------

    #[test]
    fn config_default_all_active() {
        let c = WarpConfig::default();
        assert_eq!(c.warp_size, 32);
        assert_eq!(c.active_mask, 0xFFFF_FFFF);
        assert_eq!(c.active_count(), 32);
    }

    #[test]
    fn config_new_equals_default() {
        assert_eq!(WarpConfig::new(), WarpConfig::default());
    }

    #[test]
    fn config_with_mask_valid() {
        let c = WarpConfig::with_mask(0xFF).unwrap();
        assert_eq!(c.active_count(), 8);
        assert!(c.is_active(7));
        assert!(!c.is_active(8));
    }

    #[test]
    fn config_with_mask_zero_rejected() {
        assert_eq!(WarpConfig::with_mask(0), Err(WarpError::EmptyMask));
    }

    #[test]
    fn config_is_active_out_of_range() {
        let c = full_config();
        assert!(!c.is_active(32));
        assert!(!c.is_active(100));
    }

    #[test]
    fn config_first_active_lane() {
        let c = WarpConfig::with_mask(0b1100).unwrap();
        assert_eq!(c.first_active_lane(), Some(2));
    }

    #[test]
    fn config_first_active_lane_full() {
        assert_eq!(full_config().first_active_lane(), Some(0));
    }

    #[test]
    fn config_single_lane_mask() {
        let c = WarpConfig::with_mask(1 << 20).unwrap();
        assert_eq!(c.active_count(), 1);
        assert!(c.is_active(20));
        assert!(!c.is_active(19));
    }

    // -------------------------------------------------------------------
    // WarpReduce / WarpScan / WarpShuffle enums
    // -------------------------------------------------------------------

    #[test]
    fn reduce_enum_debug() {
        assert_eq!(format!("{:?}", WarpReduce::Sum), "Sum");
        assert_eq!(format!("{:?}", WarpReduce::Product), "Product");
    }

    #[test]
    fn scan_enum_debug() {
        assert_eq!(format!("{:?}", WarpScan::InclusiveSum), "InclusiveSum");
        assert_eq!(format!("{:?}", WarpScan::InclusiveMax), "InclusiveMax");
    }

    #[test]
    fn shuffle_xor_debug() {
        let s = WarpShuffle::Xor { xor_mask: 1 };
        assert!(format!("{:?}", s).contains("Xor"));
    }

    // -------------------------------------------------------------------
    // warp_reduce_sum
    // -------------------------------------------------------------------

    #[test]
    fn reduce_sum_all_active() {
        let c = full_config();
        let mut data = iota_data();
        warp_reduce_sum(&mut data, &c).unwrap();
        let expected: f32 = (1..=32).sum::<u32>() as f32;
        for &v in &data {
            assert!((v - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn reduce_sum_partial_mask() {
        let c = WarpConfig::with_mask(0x0F).unwrap();
        let mut data = vec![0.0; 32];
        data[0] = 1.0;
        data[1] = 2.0;
        data[2] = 3.0;
        data[3] = 4.0;
        data[4] = 100.0;
        warp_reduce_sum(&mut data, &c).unwrap();
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[4] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_single_lane() {
        let c = WarpConfig::with_mask(1).unwrap();
        let mut data = vec![0.0; 32];
        data[0] = 42.0;
        warp_reduce_sum(&mut data, &c).unwrap();
        assert!((data[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_zeros() {
        let c = full_config();
        let mut data = vec![0.0; 32];
        warp_reduce_sum(&mut data, &c).unwrap();
        assert!(data.iter().all(|&v| v.abs() < 1e-7));
    }

    #[test]
    fn reduce_sum_too_short() {
        let c = full_config();
        let mut data = vec![1.0; 16];
        assert!(warp_reduce_sum(&mut data, &c).is_err());
    }

    #[test]
    fn reduce_sum_negative_values() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
        warp_reduce_sum(&mut data, &c).unwrap();
        let expected: f32 = (0..32).map(|i| -(i as f32)).sum();
        assert!((data[0] - expected).abs() < 1e-2);
    }

    // -------------------------------------------------------------------
    // warp_reduce_max
    // -------------------------------------------------------------------

    #[test]
    fn reduce_max_all_active() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        warp_reduce_max(&mut data, &c).unwrap();
        for &v in &data {
            assert!((v - 31.0).abs() < 1e-5);
        }
    }

    #[test]
    fn reduce_max_partial_mask() {
        let c = WarpConfig::with_mask(0b1010).unwrap();
        let mut data = vec![0.0; 32];
        data[0] = 999.0;
        data[1] = 3.0;
        data[3] = 7.0;
        warp_reduce_max(&mut data, &c).unwrap();
        assert!((data[1] - 7.0).abs() < 1e-5);
        assert!((data[0] - 999.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_max_negative_values() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| -100.0 + i as f32).collect();
        warp_reduce_max(&mut data, &c).unwrap();
        assert!((data[0] - (-69.0)).abs() < 1e-5);
    }

    #[test]
    fn reduce_max_too_short() {
        let c = full_config();
        let mut data = vec![1.0; 10];
        assert!(warp_reduce_max(&mut data, &c).is_err());
    }

    // -------------------------------------------------------------------
    // warp_reduce (generic)
    // -------------------------------------------------------------------

    #[test]
    fn reduce_generic_min() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| (i + 10) as f32).collect();
        warp_reduce(&mut data, WarpReduce::Min, &c).unwrap();
        for &v in &data {
            assert!((v - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn reduce_generic_product() {
        let c = WarpConfig::with_mask(0b111).unwrap();
        let mut data = vec![1.0; 32];
        data[0] = 2.0;
        data[1] = 3.0;
        data[2] = 4.0;
        warp_reduce(&mut data, WarpReduce::Product, &c).unwrap();
        assert!((data[0] - 24.0).abs() < 1e-4);
        assert!((data[1] - 24.0).abs() < 1e-4);
        assert!((data[2] - 24.0).abs() < 1e-4);
    }

    #[test]
    fn reduce_generic_sum_matches_dedicated() {
        let c = full_config();
        let mut d1 = iota_data();
        let mut d2 = iota_data();
        warp_reduce_sum(&mut d1, &c).unwrap();
        warp_reduce(&mut d2, WarpReduce::Sum, &c).unwrap();
        assert!((d1[0] - d2[0]).abs() < 1e-4);
    }

    #[test]
    fn reduce_generic_max_matches_dedicated() {
        let c = full_config();
        let mut d1 = iota_data();
        let mut d2 = iota_data();
        warp_reduce_max(&mut d1, &c).unwrap();
        warp_reduce(&mut d2, WarpReduce::Max, &c).unwrap();
        assert!((d1[0] - d2[0]).abs() < 1e-5);
    }

    // -------------------------------------------------------------------
    // warp_broadcast
    // -------------------------------------------------------------------

    #[test]
    fn broadcast_from_lane_zero() {
        let c = full_config();
        let mut data = vec![0.0; 32];
        data[0] = 42.0;
        warp_broadcast(&mut data, 0, &c).unwrap();
        for &v in &data {
            assert!((v - 42.0).abs() < 1e-7);
        }
    }

    #[test]
    fn broadcast_from_last_lane() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        warp_broadcast(&mut data, 31, &c).unwrap();
        for &v in &data {
            assert!((v - 31.0).abs() < 1e-7);
        }
    }

    #[test]
    fn broadcast_partial_mask() {
        let c = WarpConfig::with_mask(0b1111).unwrap();
        let mut data = vec![0.0; 32];
        data[2] = 10.0;
        data[10] = 99.0;
        warp_broadcast(&mut data, 2, &c).unwrap();
        assert!((data[0] - 10.0).abs() < 1e-7);
        assert!((data[10] - 99.0).abs() < 1e-7);
    }

    #[test]
    fn broadcast_inactive_source_rejected() {
        let c = WarpConfig::with_mask(1).unwrap();
        let mut data = vec![0.0; 32];
        assert!(warp_broadcast(&mut data, 1, &c).is_err());
    }

    #[test]
    fn broadcast_lane_out_of_range() {
        let c = full_config();
        let mut data = vec![0.0; 32];
        assert!(warp_broadcast(&mut data, 32, &c).is_err());
    }

    #[test]
    fn broadcast_data_too_short() {
        let c = full_config();
        let mut data = vec![1.0; 8];
        assert!(warp_broadcast(&mut data, 0, &c).is_err());
    }

    #[test]
    fn broadcast_is_idempotent() {
        let c = full_config();
        let mut data = vec![0.0; 32];
        data[0] = 5.0;
        warp_broadcast(&mut data, 0, &c).unwrap();
        let snap = data.clone();
        warp_broadcast(&mut data, 0, &c).unwrap();
        assert_eq!(data, snap);
    }

    // -------------------------------------------------------------------
    // warp_shuffle
    // -------------------------------------------------------------------

    #[test]
    fn shuffle_direct_identity() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = WarpShuffle::Direct { src_lanes: (0..32).collect() };
        let original = data.clone();
        warp_shuffle(&mut data, &s, &c).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn shuffle_direct_reverse() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = WarpShuffle::Direct { src_lanes: (0..32).rev().collect() };
        warp_shuffle(&mut data, &s, &c).unwrap();
        for i in 0..32 {
            assert!((data[i] - (31 - i) as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn shuffle_xor_swap_neighbors() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = WarpShuffle::Xor { xor_mask: 1 };
        warp_shuffle(&mut data, &s, &c).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
        assert!((data[1] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn shuffle_down_delta1() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = WarpShuffle::Down { delta: 1 };
        warp_shuffle(&mut data, &s, &c).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
        assert!((data[30] - 31.0).abs() < 1e-7);
    }

    #[test]
    fn shuffle_up_delta1() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let s = WarpShuffle::Up { delta: 1 };
        warp_shuffle(&mut data, &s, &c).unwrap();
        // lane 0 has delta > lane, so unchanged
        assert!((data[0] - 0.0).abs() < 1e-7);
        assert!((data[1] - 0.0).abs() < 1e-7);
        assert!((data[31] - 30.0).abs() < 1e-7);
    }

    #[test]
    fn shuffle_direct_src_too_short() {
        let c = full_config();
        let mut data = vec![0.0; 32];
        let s = WarpShuffle::Direct { src_lanes: vec![0; 10] };
        assert!(warp_shuffle(&mut data, &s, &c).is_err());
    }

    #[test]
    fn shuffle_direct_inactive_source_rejected() {
        let c = WarpConfig::with_mask(0b0011).unwrap();
        let mut data = vec![0.0; 32];
        let mut src: Vec<u32> = (0..32).collect();
        src[0] = 5; // inactive
        let s = WarpShuffle::Direct { src_lanes: src };
        assert!(warp_shuffle(&mut data, &s, &c).is_err());
    }

    // -------------------------------------------------------------------
    // warp_prefix_sum
    // -------------------------------------------------------------------

    #[test]
    fn prefix_sum_ones() {
        let c = full_config();
        let mut data = vec![1.0; 32];
        warp_prefix_sum(&mut data, &c).unwrap();
        for i in 0..32 {
            assert!((data[i] - (i + 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn prefix_sum_varying() {
        let c = full_config();
        let mut data = iota_data();
        warp_prefix_sum(&mut data, &c).unwrap();
        for i in 0..32 {
            let expected = ((i + 1) * (i + 2)) as f32 / 2.0;
            assert!((data[i] - expected).abs() < 1e-2);
        }
    }

    #[test]
    fn prefix_sum_partial_mask() {
        let c = WarpConfig::with_mask(0b10101).unwrap();
        let mut data = vec![0.0; 32];
        data[0] = 1.0;
        data[1] = 999.0;
        data[2] = 2.0;
        data[4] = 3.0;
        warp_prefix_sum(&mut data, &c).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 999.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[4] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn prefix_sum_too_short() {
        let c = full_config();
        let mut data = vec![1.0; 2];
        assert!(warp_prefix_sum(&mut data, &c).is_err());
    }

    #[test]
    fn prefix_sum_last_equals_reduce_sum() {
        let c = full_config();
        let vals = iota_data();
        let mut prefix = vals.clone();
        warp_prefix_sum(&mut prefix, &c).unwrap();
        let mut reduced = vals;
        warp_reduce_sum(&mut reduced, &c).unwrap();
        assert!((prefix[31] - reduced[0]).abs() < 1e-2);
    }

    // -------------------------------------------------------------------
    // warp_scan (generic)
    // -------------------------------------------------------------------

    #[test]
    fn scan_exclusive_sum_ones() {
        let c = full_config();
        let mut data = vec![1.0; 32];
        warp_scan(&mut data, WarpScan::ExclusiveSum, &c).unwrap();
        for i in 0..32 {
            assert!((data[i] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn scan_inclusive_sum_matches_prefix_sum() {
        let c = full_config();
        let mut d1 = iota_data();
        let mut d2 = iota_data();
        warp_prefix_sum(&mut d1, &c).unwrap();
        warp_scan(&mut d2, WarpScan::InclusiveSum, &c).unwrap();
        for i in 0..32 {
            assert!((d1[i] - d2[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn scan_inclusive_max() {
        let c = full_config();
        let mut data: Vec<f32> = (0..32).map(|i| (i % 5) as f32).collect();
        warp_scan(&mut data, WarpScan::InclusiveMax, &c).unwrap();
        // running max should be non-decreasing
        for i in 1..32 {
            assert!(data[i] >= data[i - 1]);
        }
    }

    #[test]
    fn scan_exclusive_first_is_zero() {
        let c = full_config();
        let mut data = iota_data();
        warp_scan(&mut data, WarpScan::ExclusiveSum, &c).unwrap();
        assert!(data[0].abs() < 1e-7);
    }

    // -------------------------------------------------------------------
    // warp_ballot
    // -------------------------------------------------------------------

    #[test]
    fn ballot_all_true() {
        let c = full_config();
        assert_eq!(warp_ballot(&vec![true; 32], &c).unwrap(), 0xFFFF_FFFF);
    }

    #[test]
    fn ballot_all_false() {
        let c = full_config();
        assert_eq!(warp_ballot(&vec![false; 32], &c).unwrap(), 0);
    }

    #[test]
    fn ballot_alternating() {
        let c = full_config();
        let preds: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        assert_eq!(warp_ballot(&preds, &c).unwrap(), 0x5555_5555);
    }

    #[test]
    fn ballot_partial_mask() {
        let c = WarpConfig::with_mask(0xFF).unwrap();
        assert_eq!(warp_ballot(&vec![true; 32], &c).unwrap(), 0xFF);
    }

    #[test]
    fn ballot_too_short() {
        let c = full_config();
        assert!(warp_ballot(&vec![true; 16], &c).is_err());
    }

    // -------------------------------------------------------------------
    // warp_all / warp_any
    // -------------------------------------------------------------------

    #[test]
    fn all_true() {
        let c = full_config();
        assert!(warp_all(&vec![true; 32], &c).unwrap());
    }

    #[test]
    fn all_one_false() {
        let c = full_config();
        let mut p = vec![true; 32];
        p[15] = false;
        assert!(!warp_all(&p, &c).unwrap());
    }

    #[test]
    fn all_partial_mask_ignores_inactive() {
        let c = WarpConfig::with_mask(0b11).unwrap();
        let mut p = vec![false; 32];
        p[0] = true;
        p[1] = true;
        assert!(warp_all(&p, &c).unwrap());
    }

    #[test]
    fn any_all_false() {
        let c = full_config();
        assert!(!warp_any(&vec![false; 32], &c).unwrap());
    }

    #[test]
    fn any_one_true() {
        let c = full_config();
        let mut p = vec![false; 32];
        p[31] = true;
        assert!(warp_any(&p, &c).unwrap());
    }

    #[test]
    fn any_partial_mask_ignores_inactive() {
        let c = WarpConfig::with_mask(1).unwrap();
        let mut p = vec![false; 32];
        p[1] = true; // inactive
        assert!(!warp_any(&p, &c).unwrap());
    }

    #[test]
    fn all_too_short() {
        let c = full_config();
        assert!(warp_all(&vec![true; 10], &c).is_err());
    }

    #[test]
    fn any_too_short() {
        let c = full_config();
        assert!(warp_any(&vec![true; 10], &c).is_err());
    }

    // -------------------------------------------------------------------
    // Consistency / integration
    // -------------------------------------------------------------------

    #[test]
    fn ballot_consistent_with_all_any() {
        let c = full_config();
        let preds = vec![true; 32];
        let ballot = warp_ballot(&preds, &c).unwrap();
        let all = warp_all(&preds, &c).unwrap();
        let any = warp_any(&preds, &c).unwrap();
        assert_eq!(ballot, 0xFFFF_FFFF);
        assert!(all);
        assert!(any);
    }

    #[test]
    fn ballot_empty_consistent_with_all_any() {
        let c = full_config();
        let preds = vec![false; 32];
        assert_eq!(warp_ballot(&preds, &c).unwrap(), 0);
        assert!(!warp_all(&preds, &c).unwrap());
        assert!(!warp_any(&preds, &c).unwrap());
    }

    #[test]
    fn shuffle_then_reduce_commutes() {
        let c = full_config();
        let orig = iota_data();
        let mut r1 = orig.clone();
        warp_reduce_sum(&mut r1, &c).unwrap();
        let mut r2 = orig;
        let s = WarpShuffle::Direct { src_lanes: (0..32).rev().collect() };
        warp_shuffle(&mut r2, &s, &c).unwrap();
        warp_reduce_sum(&mut r2, &c).unwrap();
        assert!((r1[0] - r2[0]).abs() < 1e-2);
    }

    #[test]
    fn exclusive_scan_shift_of_inclusive() {
        let c = full_config();
        let vals = vec![2.0f32; 32];
        let mut inc = vals.clone();
        warp_prefix_sum(&mut inc, &c).unwrap();
        let mut exc = vals;
        warp_scan(&mut exc, WarpScan::ExclusiveSum, &c).unwrap();
        for i in 0..32 {
            assert!((exc[i] - (inc[i] - 2.0)).abs() < 1e-5);
        }
    }

    // -------------------------------------------------------------------
    // CUDA kernel source existence (GPU-gated)
    // -------------------------------------------------------------------

    #[test]
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn cuda_kernel_source_non_empty() {
        assert!(!CUDA_WARP_OPS_SRC.is_empty());
        assert!(CUDA_WARP_OPS_SRC.contains("__shfl_xor_sync"));
    }
}

// ===========================================================================
// Property-based tests
// ===========================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn warp_f32_vec() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1000.0f32..1000.0, 32..=32)
    }

    fn active_mask_strategy() -> impl Strategy<Value = u32> {
        (1u32..=0xFFFF_FFFFu32)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Reduce-sum should equal the iterator sum of active lanes.
        #[test]
        fn prop_reduce_sum_matches_iter(data in warp_f32_vec()) {
            let c = WarpConfig::new();
            let expected: f32 = data.iter().sum();
            let mut buf = data;
            warp_reduce_sum(&mut buf, &c).unwrap();
            prop_assert!((buf[0] - expected).abs() < 1e-1,
                "reduce_sum={} expected={}", buf[0], expected);
        }

        /// Reduce-max should equal the iterator max of active lanes.
        #[test]
        fn prop_reduce_max_matches_iter(data in warp_f32_vec()) {
            let c = WarpConfig::new();
            let expected = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut buf = data;
            warp_reduce_max(&mut buf, &c).unwrap();
            prop_assert!((buf[0] - expected).abs() < 1e-5,
                "reduce_max={} expected={}", buf[0], expected);
        }

        /// Prefix-sum last element equals reduce-sum.
        #[test]
        fn prop_prefix_sum_last_eq_reduce(data in warp_f32_vec()) {
            let c = WarpConfig::new();
            let mut prefix = data.clone();
            warp_prefix_sum(&mut prefix, &c).unwrap();
            let mut red = data;
            warp_reduce_sum(&mut red, &c).unwrap();
            prop_assert!((prefix[31] - red[0]).abs() < 1e-0,
                "prefix_last={} reduce={}", prefix[31], red[0]);
        }

        /// Ballot popcount with all-true predicates equals active_count.
        #[test]
        fn prop_ballot_popcount_eq_active(mask in active_mask_strategy()) {
            let c = WarpConfig::with_mask(mask).unwrap();
            let preds = vec![true; 32];
            let ballot = warp_ballot(&preds, &c).unwrap();
            prop_assert_eq!(ballot.count_ones(), c.active_count());
        }

        /// Broadcast makes all active lanes identical to the source.
        #[test]
        fn prop_broadcast_uniform(
            data in warp_f32_vec(),
            src_lane in 0u32..32
        ) {
            let c = WarpConfig::new();
            let expected = data[src_lane as usize];
            let mut buf = data;
            warp_broadcast(&mut buf, src_lane, &c).unwrap();
            for i in 0..32u32 {
                if c.is_active(i) {
                    prop_assert!((buf[i as usize] - expected).abs() < 1e-7);
                }
            }
        }
    }
}
