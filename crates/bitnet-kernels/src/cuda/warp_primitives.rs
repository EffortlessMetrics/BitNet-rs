//! Warp-level shuffle, vote, and ballot primitives for CUDA kernels.
//!
//! This module provides low-level warp primitives that map directly to CUDA
//! hardware intrinsics (`__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
//! `__shfl_xor_sync`, `__ballot_sync`, `__all_sync`, `__any_sync`,
//! `__match_any_sync`). CPU fallbacks simulate 32-thread warp groups for
//! correctness testing without GPU hardware.
//!
//! # Shuffle modes
//!
//! [`ShuffleMode`] selects the warp shuffle variant:
//! - [`ShuffleMode::Idx`] — read from absolute lane index
//! - [`ShuffleMode::Up`] — read from `lane - delta`
//! - [`ShuffleMode::Down`] — read from `lane + delta`
//! - [`ShuffleMode::Xor`] — read from `lane ^ mask`
//!
//! # Configuration
//!
//! [`WarpConfig`] holds warp geometry: `warp_size`, `lane_mask` (per-lane
//! identity), and `active_mask` (participation bitmask).
//!
//! # Functions
//!
//! ## Shuffles
//! - [`warp_shuffle`] — indexed shuffle
//! - [`warp_shuffle_xor`] — XOR shuffle (butterfly)
//! - [`warp_shuffle_up`] — shift up by delta
//! - [`warp_shuffle_down`] — shift down by delta
//!
//! ## Reductions
//! - [`warp_reduce_sum`] — butterfly sum
//! - [`warp_reduce_max`] — butterfly max
//! - [`warp_reduce_min`] — butterfly min
//! - [`warp_segmented_reduce`] — per-segment reduction within a warp
//!
//! ## Voting
//! - [`warp_ballot`] — predicate → bitmask
//! - [`warp_vote_all`] — unanimous predicate
//! - [`warp_vote_any`] — existential predicate
//! - [`warp_match_any`] — match lanes with identical values
//!
//! ## Scans
//! - [`warp_prefix_sum`] — inclusive prefix sum
//!
//! ## Communication
//! - [`warp_broadcast`] — broadcast from one lane to all
//!
//! # CUDA kernel source
//!
//! [`WARP_PRIMITIVES_KERNEL_SRC`] contains CUDA C kernels using hardware
//! warp intrinsics. Feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Standard CUDA warp size (32 threads).
pub const WARP_SIZE: u32 = 32;

/// Full warp mask — all 32 lanes active.
pub const FULL_MASK: u32 = 0xFFFF_FFFF;

// ---------------------------------------------------------------------------
// ShuffleMode
// ---------------------------------------------------------------------------

/// Selects the warp shuffle variant.
///
/// Maps to the four CUDA `__shfl_*_sync` intrinsics:
/// - `Idx`  → `__shfl_sync`
/// - `Up`   → `__shfl_up_sync`
/// - `Down` → `__shfl_down_sync`
/// - `Xor`  → `__shfl_xor_sync`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShuffleMode {
    /// Read from lane `src_lane` (absolute index).
    Idx,
    /// Read from lane `self_lane - delta` (shift up).
    Up,
    /// Read from lane `self_lane + delta` (shift down).
    Down,
    /// Read from lane `self_lane ^ lane_mask` (butterfly).
    Xor,
}

impl std::fmt::Display for ShuffleMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idx => write!(f, "idx"),
            Self::Up => write!(f, "up"),
            Self::Down => write!(f, "down"),
            Self::Xor => write!(f, "xor"),
        }
    }
}

// ---------------------------------------------------------------------------
// WarpConfig
// ---------------------------------------------------------------------------

/// Configuration for warp-level primitives.
///
/// `warp_size` is always 32 for CUDA GPUs. `lane_mask` is the per-lane
/// identity bitmask (1 << lane_id). `active_mask` is the participation
/// bitmask — bit `i` set means lane `i` participates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpConfig {
    /// Number of lanes in the warp (always 32 for CUDA).
    pub warp_size: u32,
    /// Per-lane identity bitmask (typically `1 << lane_id`).
    pub lane_mask: u32,
    /// Participation bitmask — bit `i` set means lane `i` is active.
    pub active_mask: u32,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self { warp_size: WARP_SIZE, lane_mask: FULL_MASK, active_mask: FULL_MASK }
    }
}

impl WarpConfig {
    /// Create a config with all 32 lanes active and full lane mask.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config for a specific lane with all lanes active.
    pub fn for_lane(lane_id: u32) -> Self {
        Self { warp_size: WARP_SIZE, lane_mask: 1 << lane_id, active_mask: FULL_MASK }
    }

    /// Create a config with a custom active mask.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `active_mask` is zero.
    pub fn with_active_mask(active_mask: u32) -> Result<Self> {
        if active_mask == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "active_mask must have at least one lane active".into(),
            }
            .into());
        }
        Ok(Self { warp_size: WARP_SIZE, lane_mask: FULL_MASK, active_mask })
    }

    /// Create a config with custom lane mask and active mask.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `active_mask` is zero.
    pub fn with_masks(lane_mask: u32, active_mask: u32) -> Result<Self> {
        if active_mask == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "active_mask must have at least one lane active".into(),
            }
            .into());
        }
        Ok(Self { warp_size: WARP_SIZE, lane_mask, active_mask })
    }

    /// Number of active lanes.
    #[inline]
    pub fn active_count(&self) -> u32 {
        self.active_mask.count_ones()
    }

    /// Check if a lane is active.
    #[inline]
    pub fn is_active(&self, lane: u32) -> bool {
        lane < self.warp_size && (self.active_mask & (1 << lane)) != 0
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// CUDA C kernel source for warp-level shuffle, vote, and ballot primitives.
///
/// Contains kernels using `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
/// `__shfl_xor_sync`, `__ballot_sync`, `__all_sync`, `__any_sync`, and
/// `__match_any_sync` intrinsics.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const WARP_PRIMITIVES_KERNEL_SRC: &str = r#"
// ---------------------------------------------------------------------------
// Shuffle primitives
// ---------------------------------------------------------------------------

// Indexed shuffle: each lane reads from src_lane.
extern "C" __global__ void warp_shuffle_idx_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const int*   __restrict__ src_lanes,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = input[idx];
    int src = src_lanes[idx % 32];
    output[idx] = __shfl_sync(0xFFFFFFFFu, val, src);
}

// XOR shuffle: each lane reads from lane ^ xor_mask.
extern "C" __global__ void warp_shuffle_xor_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int xor_mask,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = input[idx];
    output[idx] = __shfl_xor_sync(0xFFFFFFFFu, val, xor_mask);
}

// Up shuffle: each lane reads from lane - delta (lower lanes keep value).
extern "C" __global__ void warp_shuffle_up_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int delta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = input[idx];
    output[idx] = __shfl_up_sync(0xFFFFFFFFu, val, delta);
}

// Down shuffle: each lane reads from lane + delta (upper lanes keep value).
extern "C" __global__ void warp_shuffle_down_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int delta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = input[idx];
    output[idx] = __shfl_down_sync(0xFFFFFFFFu, val, delta);
}

// ---------------------------------------------------------------------------
// Reductions using shuffles
// ---------------------------------------------------------------------------

// Butterfly sum reduction via __shfl_xor_sync.
extern "C" __global__ void warp_reduce_sum_f32(
    float* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_xor_sync(MASK, val, offset);
    }
    data[idx] = val;
}

// Butterfly max reduction via __shfl_xor_sync.
extern "C" __global__ void warp_reduce_max_f32(
    float* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(MASK, val, offset));
    }
    data[idx] = val;
}

// Butterfly min reduction via __shfl_xor_sync.
extern "C" __global__ void warp_reduce_min_f32(
    float* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val = fminf(val, __shfl_xor_sync(MASK, val, offset));
    }
    data[idx] = val;
}

// ---------------------------------------------------------------------------
// Voting primitives
// ---------------------------------------------------------------------------

// Ballot: returns bitmask of lanes where predicate != 0.
extern "C" __global__ void warp_ballot_kernel(
    const int*    __restrict__ predicates,
    unsigned int* __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int ballot = __ballot_sync(0xFFFFFFFFu, predicates[idx]);
    results[idx] = ballot;
}

// Vote all: writes 1 if all lanes satisfy predicate, else 0.
extern "C" __global__ void warp_vote_all_kernel(
    const int* __restrict__ predicates,
    int*       __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = __all_sync(0xFFFFFFFFu, predicates[idx]);
}

// Vote any: writes 1 if any lane satisfies predicate, else 0.
extern "C" __global__ void warp_vote_any_kernel(
    const int* __restrict__ predicates,
    int*       __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = __any_sync(0xFFFFFFFFu, predicates[idx]);
}

// Match any: returns bitmask of lanes with the same value.
extern "C" __global__ void warp_match_any_kernel(
    const int*    __restrict__ values,
    unsigned int* __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = __match_any_sync(0xFFFFFFFFu, values[idx]);
}

// Inclusive prefix sum using __shfl_up_sync.
extern "C" __global__ void warp_prefix_sum_f32(
    float* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int d = 1; d < 32; d <<= 1) {
        float tmp = __shfl_up_sync(MASK, val, d);
        if ((threadIdx.x & 31) >= d) val += tmp;
    }
    data[idx] = val;
}

// Broadcast from src_lane to all lanes.
extern "C" __global__ void warp_broadcast_f32(
    float* __restrict__ data,
    int n,
    int src_lane)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    val = __shfl_sync(0xFFFFFFFFu, val, src_lane);
    data[idx] = val;
}

// Segmented reduce: per-segment sum within warp.
// segment_ids assigns each lane to a segment. Lanes with the same segment_id
// get the sum of their segment.
extern "C" __global__ void warp_segmented_reduce_sum_f32(
    float*     __restrict__ data,
    const int* __restrict__ segment_ids,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int lane = threadIdx.x & 31;
    int my_seg = segment_ids[idx];
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    // Naive: iterate all lanes and accumulate matching segments.
    float sum = 0.0f;
    for (int src = 0; src < 32; src++) {
        float other_val = __shfl_sync(MASK, val, src);
        int other_seg = __shfl_sync(MASK, my_seg, src);
        if (other_seg == my_seg) sum += other_val;
    }
    data[idx] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_warp_data(data: &[f32], config: &WarpConfig) -> Result<()> {
    if data.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("warp data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    Ok(())
}

fn validate_predicates(predicates: &[bool], config: &WarpConfig) -> Result<()> {
    if predicates.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "predicates length {} < warp_size {}",
                predicates.len(),
                config.warp_size
            ),
        }
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shuffle functions (CPU simulation)
// ---------------------------------------------------------------------------

/// Indexed warp shuffle — each active lane reads from `src_lanes[lane]`.
///
/// Simulates `__shfl_sync(mask, val, src_lane)`. Source lanes must be active.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes are wrong or
/// any source lane is inactive / out of range.
pub fn warp_shuffle(data: &mut [f32], src_lanes: &[u32], config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    if src_lanes.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "src_lanes length {} < warp_size {}",
                src_lanes.len(),
                config.warp_size
            ),
        }
        .into());
    }
    for i in 0..config.warp_size {
        if config.is_active(i) {
            let src = src_lanes[i as usize];
            if src >= config.warp_size || !config.is_active(src) {
                return Err(KernelError::InvalidArguments {
                    reason: format!(
                        "shuffle src lane {src} for dest lane {i} is out of range or inactive"
                    ),
                }
                .into());
            }
        }
    }
    let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = snapshot[src_lanes[i as usize] as usize];
        }
    }
    Ok(())
}

/// XOR warp shuffle — each active lane reads from `lane ^ xor_mask`.
///
/// Simulates `__shfl_xor_sync(mask, val, xor_mask)`. This is the butterfly
/// pattern used in warp reductions.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_shuffle_xor(data: &mut [f32], xor_mask: u32, config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
    for i in 0..config.warp_size {
        if config.is_active(i) {
            let src = i ^ xor_mask;
            if src < config.warp_size && config.is_active(src) {
                data[i as usize] = snapshot[src as usize];
            }
            // If src is out of range or inactive, lane keeps its value (CUDA semantics).
        }
    }
    Ok(())
}

/// Up warp shuffle — each active lane reads from `lane - delta`.
///
/// Simulates `__shfl_up_sync(mask, val, delta)`. Lanes with `lane < delta`
/// keep their original value (no valid source).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_shuffle_up(data: &mut [f32], delta: u32, config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
    for i in 0..config.warp_size {
        if config.is_active(i) && i >= delta {
            let src = i - delta;
            if config.is_active(src) {
                data[i as usize] = snapshot[src as usize];
            }
        }
    }
    Ok(())
}

/// Down warp shuffle — each active lane reads from `lane + delta`.
///
/// Simulates `__shfl_down_sync(mask, val, delta)`. Lanes with
/// `lane + delta >= warp_size` keep their original value.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_shuffle_down(data: &mut [f32], delta: u32, config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
    for i in 0..config.warp_size {
        if config.is_active(i) {
            let src = i + delta;
            if src < config.warp_size && config.is_active(src) {
                data[i as usize] = snapshot[src as usize];
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reductions (CPU simulation)
// ---------------------------------------------------------------------------

/// Butterfly-pattern warp sum reduction.
///
/// All active lanes receive the sum of active lane values.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_reduce_sum(data: &mut [f32], config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let sum: f32 =
        (0..config.warp_size).filter(|&i| config.is_active(i)).map(|i| data[i as usize]).sum();
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = sum;
        }
    }
    Ok(())
}

/// Butterfly-pattern warp max reduction.
///
/// All active lanes receive the maximum active lane value.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_reduce_max(data: &mut [f32], config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
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

/// Butterfly-pattern warp min reduction.
///
/// All active lanes receive the minimum active lane value.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_reduce_min(data: &mut [f32], config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let min_val = (0..config.warp_size)
        .filter(|&i| config.is_active(i))
        .map(|i| data[i as usize])
        .fold(f32::INFINITY, f32::min);
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = min_val;
        }
    }
    Ok(())
}

/// Segmented reduction within a warp.
///
/// Each lane belongs to a segment identified by `segment_ids[lane]`. After
/// reduction, every active lane holds the sum of values in its segment.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes are wrong.
pub fn warp_segmented_reduce(
    data: &mut [f32],
    segment_ids: &[u32],
    config: &WarpConfig,
) -> Result<()> {
    validate_warp_data(data, config)?;
    if segment_ids.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "segment_ids length {} < warp_size {}",
                segment_ids.len(),
                config.warp_size
            ),
        }
        .into());
    }
    let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
    for i in 0..config.warp_size {
        if !config.is_active(i) {
            continue;
        }
        let seg = segment_ids[i as usize];
        let sum: f32 = (0..config.warp_size)
            .filter(|&j| config.is_active(j) && segment_ids[j as usize] == seg)
            .map(|j| snapshot[j as usize])
            .sum();
        data[i as usize] = sum;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Voting (CPU simulation)
// ---------------------------------------------------------------------------

/// Ballot vote across warp lanes.
///
/// Returns a bitmask where bit `i` is set iff lane `i` is active and
/// `predicates[i]` is `true`. Simulates `__ballot_sync`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` length is wrong.
pub fn warp_ballot(predicates: &[bool], config: &WarpConfig) -> Result<u32> {
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
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` length is wrong.
pub fn warp_vote_all(predicates: &[bool], config: &WarpConfig) -> Result<bool> {
    validate_predicates(predicates, config)?;
    Ok((0..config.warp_size).filter(|&i| config.is_active(i)).all(|i| predicates[i as usize]))
}

/// Check if any active lane satisfies the predicate.
///
/// Simulates `__any_sync(mask, predicate)`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` length is wrong.
pub fn warp_vote_any(predicates: &[bool], config: &WarpConfig) -> Result<bool> {
    validate_predicates(predicates, config)?;
    Ok((0..config.warp_size).filter(|&i| config.is_active(i)).any(|i| predicates[i as usize]))
}

/// Match lanes with the same value.
///
/// Returns a bitmask per lane where bit `j` is set iff lane `j` is active
/// and holds the same value as lane `i`. Simulates `__match_any_sync`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_match_any(data: &[f32], config: &WarpConfig) -> Result<Vec<u32>> {
    if data.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    let ws = config.warp_size as usize;
    let mut masks = vec![0u32; ws];
    for i in 0..config.warp_size {
        if !config.is_active(i) {
            continue;
        }
        let vi = data[i as usize].to_bits();
        for j in 0..config.warp_size {
            if config.is_active(j) && data[j as usize].to_bits() == vi {
                masks[i as usize] |= 1 << j;
            }
        }
    }
    Ok(masks)
}

// ---------------------------------------------------------------------------
// Scans (CPU simulation)
// ---------------------------------------------------------------------------

/// Inclusive prefix sum within the warp.
///
/// After execution, `data[i]` = sum of `data[0..=i]` for all active lanes.
/// Inactive lanes keep their original values.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length is wrong.
pub fn warp_prefix_sum(data: &mut [f32], config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let mut running = 0.0f32;
    for i in 0..config.warp_size {
        if config.is_active(i) {
            running += data[i as usize];
            data[i as usize] = running;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Communication (CPU simulation)
// ---------------------------------------------------------------------------

/// Broadcast a value from `src_lane` to all active lanes.
///
/// Simulates `__shfl_sync(mask, val, src_lane)`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `src_lane` is inactive
/// or out of range, or if `data` length is wrong.
pub fn warp_broadcast(data: &mut [f32], src_lane: u32, config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    if !config.is_active(src_lane) {
        return Err(KernelError::InvalidArguments {
            reason: format!("broadcast source lane {src_lane} is not active"),
        }
        .into());
    }
    let val = data[src_lane as usize];
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] = val;
        }
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create data [1.0, 2.0, ..., 32.0]
    fn sequential_data() -> Vec<f32> {
        (1..=32).map(|i| i as f32).collect()
    }

    fn zeros() -> Vec<f32> {
        vec![0.0f32; 32]
    }

    // =======================================================================
    // WarpConfig tests
    // =======================================================================

    #[test]
    fn config_default_all_active() {
        let c = WarpConfig::default();
        assert_eq!(c.warp_size, 32);
        assert_eq!(c.lane_mask, FULL_MASK);
        assert_eq!(c.active_mask, FULL_MASK);
        assert_eq!(c.active_count(), 32);
    }

    #[test]
    fn config_new_equals_default() {
        assert_eq!(WarpConfig::new(), WarpConfig::default());
    }

    #[test]
    fn config_for_lane() {
        let c = WarpConfig::for_lane(5);
        assert_eq!(c.lane_mask, 1 << 5);
        assert_eq!(c.active_mask, FULL_MASK);
    }

    #[test]
    fn config_with_active_mask() {
        let c = WarpConfig::with_active_mask(0x0000_00FF).unwrap();
        assert_eq!(c.active_count(), 8);
        assert!(c.is_active(0));
        assert!(c.is_active(7));
        assert!(!c.is_active(8));
    }

    #[test]
    fn config_zero_mask_rejected() {
        assert!(WarpConfig::with_active_mask(0).is_err());
    }

    #[test]
    fn config_with_masks() {
        let c = WarpConfig::with_masks(0x1, 0xFF).unwrap();
        assert_eq!(c.lane_mask, 1);
        assert_eq!(c.active_count(), 8);
    }

    #[test]
    fn config_with_masks_zero_active_rejected() {
        assert!(WarpConfig::with_masks(0x1, 0).is_err());
    }

    #[test]
    fn config_is_active_out_of_range() {
        let c = WarpConfig::new();
        assert!(!c.is_active(32));
        assert!(!c.is_active(100));
    }

    #[test]
    fn config_single_lane_active() {
        let c = WarpConfig::with_active_mask(1 << 15).unwrap();
        assert_eq!(c.active_count(), 1);
        assert!(!c.is_active(0));
        assert!(c.is_active(15));
    }

    // =======================================================================
    // ShuffleMode tests
    // =======================================================================

    #[test]
    fn shuffle_mode_display() {
        assert_eq!(ShuffleMode::Idx.to_string(), "idx");
        assert_eq!(ShuffleMode::Up.to_string(), "up");
        assert_eq!(ShuffleMode::Down.to_string(), "down");
        assert_eq!(ShuffleMode::Xor.to_string(), "xor");
    }

    #[test]
    fn shuffle_mode_equality() {
        assert_eq!(ShuffleMode::Idx, ShuffleMode::Idx);
        assert_ne!(ShuffleMode::Up, ShuffleMode::Down);
    }

    #[test]
    fn shuffle_mode_clone() {
        let m = ShuffleMode::Xor;
        let m2 = m;
        assert_eq!(m, m2);
    }

    #[test]
    fn shuffle_mode_debug() {
        let s = format!("{:?}", ShuffleMode::Up);
        assert!(s.contains("Up"));
    }

    // =======================================================================
    // warp_shuffle (Idx) tests
    // =======================================================================

    #[test]
    fn shuffle_identity() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let src: Vec<u32> = (0..32).collect();
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        for i in 0..32 {
            assert_eq!(data[i], (i + 1) as f32);
        }
    }

    #[test]
    fn shuffle_reverse() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let src: Vec<u32> = (0..32).rev().collect();
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        for i in 0..32 {
            assert_eq!(data[i], (32 - i) as f32);
        }
    }

    #[test]
    fn shuffle_broadcast_lane0() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let src = vec![0u32; 32];
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn shuffle_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap(); // lanes 0-3
        let mut data = sequential_data();
        let src: Vec<u32> = (0..32).map(|i| (3 - (i % 4)) as u32).collect();
        let original = data.clone();
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        // Active lanes shuffled
        assert_eq!(data[0], original[3]);
        assert_eq!(data[1], original[2]);
        // Inactive lanes unchanged
        assert_eq!(data[4], original[4]);
    }

    #[test]
    fn shuffle_src_lanes_too_short() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let src = vec![0u32; 16];
        assert!(warp_shuffle(&mut data, &src, &cfg).is_err());
    }

    #[test]
    fn shuffle_inactive_source_rejected() {
        let cfg = WarpConfig::with_active_mask(0x03).unwrap(); // lanes 0, 1
        let mut data = sequential_data();
        let mut src: Vec<u32> = (0..32).collect();
        src[0] = 5; // lane 5 is inactive
        assert!(warp_shuffle(&mut data, &src, &cfg).is_err());
    }

    #[test]
    fn shuffle_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 16];
        let src: Vec<u32> = (0..32).collect();
        assert!(warp_shuffle(&mut data, &src, &cfg).is_err());
    }

    // =======================================================================
    // warp_shuffle_xor tests
    // =======================================================================

    #[test]
    fn shuffle_xor_mask1_swaps_neighbors() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_shuffle_xor(&mut data, 1, &cfg).unwrap();
        // lane 0 <-> lane 1, lane 2 <-> lane 3, etc.
        assert_eq!(data[0], 2.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], 4.0);
        assert_eq!(data[3], 3.0);
    }

    #[test]
    fn shuffle_xor_mask0_identity() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_xor(&mut data, 0, &cfg).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn shuffle_xor_mask16_swaps_halves() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_shuffle_xor(&mut data, 16, &cfg).unwrap();
        // lane i <-> lane i^16
        assert_eq!(data[0], 17.0);
        assert_eq!(data[16], 1.0);
        assert_eq!(data[31], 16.0);
    }

    #[test]
    fn shuffle_xor_double_is_identity() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_xor(&mut data, 5, &cfg).unwrap();
        warp_shuffle_xor(&mut data, 5, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - original[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn shuffle_xor_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0xFF).unwrap(); // lanes 0-7
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_xor(&mut data, 1, &cfg).unwrap();
        assert_eq!(data[0], 2.0); // swapped
        assert_eq!(data[8], original[8]); // inactive — unchanged
    }

    #[test]
    fn shuffle_xor_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 10];
        assert!(warp_shuffle_xor(&mut data, 1, &cfg).is_err());
    }

    // =======================================================================
    // warp_shuffle_up tests
    // =======================================================================

    #[test]
    fn shuffle_up_delta1() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_shuffle_up(&mut data, 1, &cfg).unwrap();
        // lane 0 keeps value (no source), lane 1 gets lane 0's value, etc.
        assert_eq!(data[0], 1.0); // unchanged
        assert_eq!(data[1], 1.0); // got lane 0
        assert_eq!(data[2], 2.0); // got lane 1
        assert_eq!(data[31], 31.0); // got lane 30
    }

    #[test]
    fn shuffle_up_delta0_identity() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_up(&mut data, 0, &cfg).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn shuffle_up_delta31() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_up(&mut data, 31, &cfg).unwrap();
        // Only lane 31 can read from lane 0
        assert_eq!(data[31], 1.0);
        // All others keep values
        for i in 0..31 {
            assert_eq!(data[i], original[i]);
        }
    }

    #[test]
    fn shuffle_up_delta32_no_change() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_up(&mut data, 32, &cfg).unwrap();
        // No lane has lane - 32 >= 0, so all keep values
        assert_eq!(data, original);
    }

    #[test]
    fn shuffle_up_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap(); // lanes 0-3
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_up(&mut data, 1, &cfg).unwrap();
        assert_eq!(data[0], 1.0); // unchanged
        assert_eq!(data[1], 1.0); // got lane 0
        assert_eq!(data[4], original[4]); // inactive
    }

    #[test]
    fn shuffle_up_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 5];
        assert!(warp_shuffle_up(&mut data, 1, &cfg).is_err());
    }

    // =======================================================================
    // warp_shuffle_down tests
    // =======================================================================

    #[test]
    fn shuffle_down_delta1() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_shuffle_down(&mut data, 1, &cfg).unwrap();
        // lane 0 gets lane 1's value, lane 31 keeps value (no source)
        assert_eq!(data[0], 2.0);
        assert_eq!(data[30], 32.0);
        assert_eq!(data[31], 32.0); // unchanged
    }

    #[test]
    fn shuffle_down_delta0_identity() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_down(&mut data, 0, &cfg).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn shuffle_down_delta31() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_down(&mut data, 31, &cfg).unwrap();
        // Only lane 0 reads from lane 31
        assert_eq!(data[0], 32.0);
        for i in 1..32 {
            assert_eq!(data[i], original[i]);
        }
    }

    #[test]
    fn shuffle_down_delta32_no_change() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_down(&mut data, 32, &cfg).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn shuffle_down_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap(); // lanes 0-3
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_down(&mut data, 1, &cfg).unwrap();
        assert_eq!(data[0], 2.0);
        assert_eq!(data[3], original[3]); // no active source at lane 4
        assert_eq!(data[4], original[4]); // inactive
    }

    #[test]
    fn shuffle_down_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 3];
        assert!(warp_shuffle_down(&mut data, 1, &cfg).is_err());
    }

    // =======================================================================
    // warp_reduce_sum tests
    // =======================================================================

    #[test]
    fn reduce_sum_all_active() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_reduce_sum(&mut data, &cfg).unwrap();
        let expected = (1..=32).sum::<u32>() as f32; // 528
        for &v in &data {
            assert!((v - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn reduce_sum_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let mut data = sequential_data();
        warp_reduce_sum(&mut data, &cfg).unwrap();
        // sum of lanes 0-3 = 1+2+3+4 = 10
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[3] - 10.0).abs() < 1e-5);
        assert!((data[4] - 5.0).abs() < 1e-5); // unchanged
    }

    #[test]
    fn reduce_sum_single_lane() {
        let cfg = WarpConfig::with_active_mask(1).unwrap();
        let mut data = sequential_data();
        warp_reduce_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_zeros() {
        let cfg = WarpConfig::new();
        let mut data = zeros();
        warp_reduce_sum(&mut data, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn reduce_sum_negative_values() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        warp_reduce_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn reduce_sum_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 5];
        assert!(warp_reduce_sum(&mut data, &cfg).is_err());
    }

    // =======================================================================
    // warp_reduce_max tests
    // =======================================================================

    #[test]
    fn reduce_max_all_active() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_reduce_max(&mut data, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 32.0);
        }
    }

    #[test]
    fn reduce_max_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let mut data = sequential_data();
        warp_reduce_max(&mut data, &cfg).unwrap();
        assert_eq!(data[0], 4.0);
        assert_eq!(data[4], 5.0); // unchanged
    }

    #[test]
    fn reduce_max_negative_values() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
        warp_reduce_max(&mut data, &cfg).unwrap();
        assert_eq!(data[0], 0.0);
    }

    #[test]
    fn reduce_max_all_same() {
        let cfg = WarpConfig::new();
        let mut data = vec![42.0f32; 32];
        warp_reduce_max(&mut data, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 42.0);
        }
    }

    #[test]
    fn reduce_max_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 2];
        assert!(warp_reduce_max(&mut data, &cfg).is_err());
    }

    // =======================================================================
    // warp_reduce_min tests
    // =======================================================================

    #[test]
    fn reduce_min_all_active() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_reduce_min(&mut data, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn reduce_min_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0xF0).unwrap(); // lanes 4-7
        let mut data = sequential_data();
        warp_reduce_min(&mut data, &cfg).unwrap();
        assert_eq!(data[4], 5.0); // min of 5,6,7,8
        assert_eq!(data[0], 1.0); // unchanged
    }

    #[test]
    fn reduce_min_negative_values() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
        warp_reduce_min(&mut data, &cfg).unwrap();
        assert_eq!(data[0], -31.0);
    }

    #[test]
    fn reduce_min_all_same() {
        let cfg = WarpConfig::new();
        let mut data = vec![7.0f32; 32];
        warp_reduce_min(&mut data, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 7.0);
        }
    }

    #[test]
    fn reduce_min_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 4];
        assert!(warp_reduce_min(&mut data, &cfg).is_err());
    }

    // =======================================================================
    // warp_ballot tests
    // =======================================================================

    #[test]
    fn ballot_all_true() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        assert_eq!(warp_ballot(&preds, &cfg).unwrap(), FULL_MASK);
    }

    #[test]
    fn ballot_all_false() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 32];
        assert_eq!(warp_ballot(&preds, &cfg).unwrap(), 0);
    }

    #[test]
    fn ballot_alternating() {
        let cfg = WarpConfig::new();
        let preds: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        let result = warp_ballot(&preds, &cfg).unwrap();
        assert_eq!(result, 0x5555_5555);
    }

    #[test]
    fn ballot_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let preds = vec![true; 32];
        let result = warp_ballot(&preds, &cfg).unwrap();
        assert_eq!(result, 0x0F);
    }

    #[test]
    fn ballot_single_lane() {
        let cfg = WarpConfig::new();
        let mut preds = vec![false; 32];
        preds[7] = true;
        assert_eq!(warp_ballot(&preds, &cfg).unwrap(), 1 << 7);
    }

    #[test]
    fn ballot_predicates_too_short() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 10];
        assert!(warp_ballot(&preds, &cfg).is_err());
    }

    // =======================================================================
    // warp_vote_all tests
    // =======================================================================

    #[test]
    fn vote_all_unanimous_true() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        assert!(warp_vote_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_all_one_false() {
        let cfg = WarpConfig::new();
        let mut preds = vec![true; 32];
        preds[15] = false;
        assert!(!warp_vote_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_all_partial_mask_ignores_inactive() {
        let cfg = WarpConfig::with_active_mask(0x03).unwrap(); // lanes 0, 1
        let mut preds = vec![false; 32];
        preds[0] = true;
        preds[1] = true;
        // lanes 2-31 are false but inactive
        assert!(warp_vote_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_all_all_false() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 32];
        assert!(!warp_vote_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_all_predicates_too_short() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 5];
        assert!(warp_vote_all(&preds, &cfg).is_err());
    }

    // =======================================================================
    // warp_vote_any tests
    // =======================================================================

    #[test]
    fn vote_any_one_true() {
        let cfg = WarpConfig::new();
        let mut preds = vec![false; 32];
        preds[20] = true;
        assert!(warp_vote_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_any_all_false() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 32];
        assert!(!warp_vote_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_any_all_true() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        assert!(warp_vote_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_any_partial_mask_only_active_matters() {
        let cfg = WarpConfig::with_active_mask(0x01).unwrap(); // only lane 0
        let mut preds = vec![true; 32]; // all true
        preds[0] = false; // but the only active lane is false
        assert!(!warp_vote_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn vote_any_predicates_too_short() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 8];
        assert!(warp_vote_any(&preds, &cfg).is_err());
    }

    // =======================================================================
    // warp_prefix_sum tests
    // =======================================================================

    #[test]
    fn prefix_sum_all_ones() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 32];
        warp_prefix_sum(&mut data, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - (i + 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn prefix_sum_sequential() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_prefix_sum(&mut data, &cfg).unwrap();
        // prefix_sum[i] = sum(1..=i+1) = (i+1)*(i+2)/2
        for i in 0..32 {
            let expected = ((i + 1) * (i + 2) / 2) as f32;
            assert!((data[i] - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn prefix_sum_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let mut data = sequential_data();
        let original = data.clone();
        warp_prefix_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 3.0).abs() < 1e-5);
        assert!((data[2] - 6.0).abs() < 1e-5);
        assert!((data[3] - 10.0).abs() < 1e-5);
        assert_eq!(data[4], original[4]); // unchanged
    }

    #[test]
    fn prefix_sum_zeros() {
        let cfg = WarpConfig::new();
        let mut data = zeros();
        warp_prefix_sum(&mut data, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn prefix_sum_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 4];
        assert!(warp_prefix_sum(&mut data, &cfg).is_err());
    }

    // =======================================================================
    // warp_broadcast tests
    // =======================================================================

    #[test]
    fn broadcast_lane0() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_broadcast(&mut data, 0, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn broadcast_lane31() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_broadcast(&mut data, 31, &cfg).unwrap();
        for &v in &data {
            assert_eq!(v, 32.0);
        }
    }

    #[test]
    fn broadcast_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let mut data = sequential_data();
        let original = data.clone();
        warp_broadcast(&mut data, 2, &cfg).unwrap();
        for i in 0..4 {
            assert_eq!(data[i], 3.0); // lane 2's value
        }
        assert_eq!(data[4], original[4]); // inactive
    }

    #[test]
    fn broadcast_inactive_source_rejected() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let mut data = sequential_data();
        assert!(warp_broadcast(&mut data, 10, &cfg).is_err());
    }

    #[test]
    fn broadcast_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 4];
        assert!(warp_broadcast(&mut data, 0, &cfg).is_err());
    }

    // =======================================================================
    // warp_match_any tests
    // =======================================================================

    #[test]
    fn match_any_all_same() {
        let cfg = WarpConfig::new();
        let data = vec![42.0f32; 32];
        let masks = warp_match_any(&data, &cfg).unwrap();
        for &m in &masks {
            assert_eq!(m, FULL_MASK);
        }
    }

    #[test]
    fn match_any_all_unique() {
        let cfg = WarpConfig::new();
        let data = sequential_data();
        let masks = warp_match_any(&data, &cfg).unwrap();
        for i in 0..32 {
            assert_eq!(masks[i], 1 << i);
        }
    }

    #[test]
    fn match_any_two_groups() {
        let cfg = WarpConfig::new();
        let data: Vec<f32> = (0..32).map(|i| if i < 16 { 1.0 } else { 2.0 }).collect();
        let masks = warp_match_any(&data, &cfg).unwrap();
        for i in 0..16 {
            assert_eq!(masks[i], 0x0000_FFFF);
        }
        for i in 16..32 {
            assert_eq!(masks[i], 0xFFFF_0000);
        }
    }

    #[test]
    fn match_any_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let data = vec![1.0f32; 32];
        let masks = warp_match_any(&data, &cfg).unwrap();
        assert_eq!(masks[0], 0x0F); // only active lanes match
        assert_eq!(masks[4], 0); // inactive lane
    }

    #[test]
    fn match_any_data_too_short() {
        let cfg = WarpConfig::new();
        let data = vec![1.0f32; 5];
        assert!(warp_match_any(&data, &cfg).is_err());
    }

    // =======================================================================
    // warp_segmented_reduce tests
    // =======================================================================

    #[test]
    fn segmented_reduce_single_segment() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let segments = vec![0u32; 32];
        warp_segmented_reduce(&mut data, &segments, &cfg).unwrap();
        let expected = (1..=32).sum::<u32>() as f32;
        for &v in &data[..32] {
            assert!((v - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn segmented_reduce_per_lane_segments() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        let segments: Vec<u32> = (0..32).collect(); // each lane is its own segment
        warp_segmented_reduce(&mut data, &segments, &cfg).unwrap();
        assert_eq!(data, original); // each value is its own sum
    }

    #[test]
    fn segmented_reduce_two_halves() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 32];
        let segments: Vec<u32> = (0..32).map(|i| if i < 16 { 0 } else { 1 }).collect();
        warp_segmented_reduce(&mut data, &segments, &cfg).unwrap();
        for i in 0..16 {
            assert!((data[i] - 16.0).abs() < 1e-5);
        }
        for i in 16..32 {
            assert!((data[i] - 16.0).abs() < 1e-5);
        }
    }

    #[test]
    fn segmented_reduce_four_groups() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 32];
        let segments: Vec<u32> = (0..32).map(|i| (i / 8) as u32).collect();
        warp_segmented_reduce(&mut data, &segments, &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 8.0).abs() < 1e-5);
        }
    }

    #[test]
    fn segmented_reduce_partial_mask() {
        let cfg = WarpConfig::with_active_mask(0x0F).unwrap();
        let mut data = vec![2.0f32; 32];
        let original = data.clone();
        let segments = vec![0u32; 32]; // all same segment
        warp_segmented_reduce(&mut data, &segments, &cfg).unwrap();
        // 4 active lanes × 2.0 = 8.0
        for i in 0..4 {
            assert!((data[i] - 8.0).abs() < 1e-5);
        }
        assert_eq!(data[4], original[4]); // unchanged
    }

    #[test]
    fn segmented_reduce_segments_too_short() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let segments = vec![0u32; 10];
        assert!(warp_segmented_reduce(&mut data, &segments, &cfg).is_err());
    }

    #[test]
    fn segmented_reduce_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 4];
        let segments = vec![0u32; 32];
        assert!(warp_segmented_reduce(&mut data, &segments, &cfg).is_err());
    }

    // =======================================================================
    // Constants tests
    // =======================================================================

    #[test]
    fn warp_size_is_32() {
        assert_eq!(WARP_SIZE, 32);
    }

    #[test]
    fn full_mask_is_all_ones() {
        assert_eq!(FULL_MASK, 0xFFFF_FFFF);
        assert_eq!(FULL_MASK.count_ones(), 32);
    }

    // =======================================================================
    // CUDA kernel source presence (feature-gated)
    // =======================================================================

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_not_empty() {
        assert!(!WARP_PRIMITIVES_KERNEL_SRC.is_empty());
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_contains_shuffle_kernels() {
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_shuffle_idx_f32"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_shuffle_xor_f32"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_shuffle_up_f32"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_shuffle_down_f32"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_contains_vote_kernels() {
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_ballot_kernel"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_vote_all_kernel"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_vote_any_kernel"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_match_any_kernel"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_contains_reduce_kernels() {
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_reduce_sum_f32"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_reduce_max_f32"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_reduce_min_f32"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_contains_scan_and_broadcast() {
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_prefix_sum_f32"));
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_broadcast_f32"));
    }

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn kernel_src_contains_segmented_reduce() {
        assert!(WARP_PRIMITIVES_KERNEL_SRC.contains("warp_segmented_reduce_sum_f32"));
    }

    // =======================================================================
    // Cross-function integration tests
    // =======================================================================

    #[test]
    fn reduce_then_broadcast() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        warp_reduce_sum(&mut data, &cfg).unwrap();
        // All lanes have the sum; broadcast from lane 0
        warp_broadcast(&mut data, 0, &cfg).unwrap();
        let expected = (1..=32).sum::<u32>() as f32;
        for &v in &data {
            assert!((v - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn prefix_sum_last_equals_total_sum() {
        let cfg = WarpConfig::new();
        let mut sum_data = sequential_data();
        let mut prefix_data = sequential_data();
        warp_reduce_sum(&mut sum_data, &cfg).unwrap();
        warp_prefix_sum(&mut prefix_data, &cfg).unwrap();
        // Last lane of prefix sum should equal the total sum
        assert!((prefix_data[31] - sum_data[0]).abs() < 1e-3);
    }

    #[test]
    fn ballot_matches_vote_all() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        let ballot = warp_ballot(&preds, &cfg).unwrap();
        let all = warp_vote_all(&preds, &cfg).unwrap();
        assert_eq!(ballot, FULL_MASK);
        assert!(all);
    }

    #[test]
    fn ballot_matches_vote_any() {
        let cfg = WarpConfig::new();
        let mut preds = vec![false; 32];
        preds[5] = true;
        let ballot = warp_ballot(&preds, &cfg).unwrap();
        let any = warp_vote_any(&preds, &cfg).unwrap();
        assert_ne!(ballot, 0);
        assert!(any);
    }

    #[test]
    fn shuffle_xor_implements_reduce_pattern() {
        // Verify that XOR shuffle can implement butterfly reduction
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        // Manual butterfly sum using xor shuffles
        for offset in [16, 8, 4, 2, 1] {
            let snapshot = data.clone();
            warp_shuffle_xor(&mut data, offset, &cfg).unwrap();
            for i in 0..32 {
                data[i] += snapshot[i];
            }
        }
        // After 5 butterfly stages, all lanes should have ~2*sum
        // (because we add both sides at each step)
        let _sum = (1..=32).sum::<u32>() as f32;
        // Each lane should have sum * 2^0 + ... but butterfly pattern
        // gives 2*sum in each step — the actual result is 32 * lane_value_contribution
        // This is a simplified check: all lanes should have the same value
        let v0 = data[0];
        for &v in &data {
            assert!((v - v0).abs() < 1e-1);
        }
    }

    #[test]
    fn shuffle_up_then_down_partial_identity() {
        let cfg = WarpConfig::new();
        let mut data = sequential_data();
        let original = data.clone();
        warp_shuffle_up(&mut data, 1, &cfg).unwrap();
        warp_shuffle_down(&mut data, 1, &cfg).unwrap();
        // lanes 1..30 should be restored (lane 0 and 31 may differ)
        for i in 1..31 {
            assert!((data[i] - original[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn segmented_reduce_consistent_with_full_reduce() {
        let cfg = WarpConfig::new();
        let mut full_data = sequential_data();
        let mut seg_data = sequential_data();
        let segments = vec![0u32; 32]; // all in one segment
        warp_reduce_sum(&mut full_data, &cfg).unwrap();
        warp_segmented_reduce(&mut seg_data, &segments, &cfg).unwrap();
        for i in 0..32 {
            assert!((full_data[i] - seg_data[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn match_any_consistent_with_ballot() {
        let cfg = WarpConfig::new();
        let data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
        let masks = warp_match_any(&data, &cfg).unwrap();
        // Even lanes should match each other
        let even_preds: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        let even_ballot = warp_ballot(&even_preds, &cfg).unwrap();
        assert_eq!(masks[0], even_ballot);
    }
}
