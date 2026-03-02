//! Warp-level primitives for efficient GPU computation.
//!
//! This module provides CPU-simulated warp-level and block-level primitives
//! that mirror CUDA warp intrinsics. On GPU these map to hardware-accelerated
//! `__shfl_sync`, `__shfl_xor_sync`, `__ballot_sync`, `__all_sync`, and
//! `__any_sync` intrinsics. The CPU fallback performs equivalent sequential
//! simulation for correctness testing and non-GPU environments.
//!
//! # Warp-level primitives
//!
//! - [`warp_reduce_sum`] — butterfly-pattern sum reduction across lanes
//! - [`warp_reduce_max`] / [`warp_reduce_min`] — max/min reduction
//! - [`warp_broadcast`] — broadcast from one lane to all
//! - [`warp_shuffle`] — direct indexed shuffle between lanes
//! - [`warp_prefix_sum`] — inclusive prefix (scan) sum
//! - [`warp_exclusive_scan`] — exclusive prefix scan
//! - [`warp_ballot`] — ballot vote (predicate → bitmask)
//! - [`warp_all`] / [`warp_any`] — unanimous / existential predicate
//! - [`warp_match`] — match lanes with identical values
//!
//! # Block-level primitives
//!
//! - [`block_reduce_sum`] — block-wide sum using warp reductions
//! - [`block_reduce_max`] — block-wide max using warp reductions
//!
//! # Composite operations
//!
//! - [`cooperative_softmax`] — numerically stable softmax using warp ops
//!
//! # CUDA kernel source
//!
//! [`WARP_OPS_KERNEL_SRC`] contains CUDA C kernels that use hardware
//! warp intrinsics for the same operations. Feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source — warp intrinsics
// ---------------------------------------------------------------------------

/// CUDA C kernel source implementing warp-level primitives.
///
/// Contains kernels for warp reductions, shuffles, ballots, and cooperative
/// softmax using `__shfl_xor_sync`, `__shfl_sync`, `__ballot_sync`,
/// `__all_sync`, and `__any_sync` intrinsics.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const WARP_OPS_KERNEL_SRC: &str = r#"
// Warp-level sum reduction via butterfly pattern.
// Each thread holds one value; after reduction lane 0 holds the sum.
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

// Warp-level max reduction via butterfly pattern.
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

// Warp-level broadcast: copy value from src_lane to all lanes.
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

// Cooperative softmax using warp-level reductions.
// One warp per row; each lane processes a strided slice.
extern "C" __global__ void cooperative_softmax_f32(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int n_rows,
    int n_cols)
{
    const int row   = blockIdx.x;
    const int lane  = threadIdx.x & 31;
    if (row >= n_rows) return;
    const int off = row * n_cols;

    // Phase 1: find row max
    float local_max = -1e38f;
    for (int c = lane; c < n_cols; c += 32) {
        float v = input[off + c];
        if (v > local_max) local_max = v;
    }
    const unsigned MASK = 0xFFFFFFFFu;
    for (int o = 16; o >= 1; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(MASK, local_max, o));

    // Phase 2: exp sum
    float local_sum = 0.0f;
    for (int c = lane; c < n_cols; c += 32) {
        local_sum += expf(input[off + c] - local_max);
    }
    for (int o = 16; o >= 1; o >>= 1)
        local_sum += __shfl_xor_sync(MASK, local_sum, o);

    // Phase 3: normalize
    float inv_sum = 1.0f / local_sum;
    for (int c = lane; c < n_cols; c += 32) {
        output[off + c] = expf(input[off + c] - local_max) * inv_sum;
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default warp size for CUDA-compatible GPUs.
pub const DEFAULT_WARP_SIZE: u32 = 32;

/// Configuration for warp-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpConfig {
    /// Number of lanes in the warp (always 32 for CUDA).
    pub warp_size: u32,
    /// Active lane mask. Bit `i` set means lane `i` participates.
    pub active_mask: u32,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self { warp_size: DEFAULT_WARP_SIZE, active_mask: 0xFFFF_FFFF }
    }
}

impl WarpConfig {
    /// Create a config with all lanes active.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config with a custom active lane mask.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if the mask is zero.
    pub fn with_mask(active_mask: u32) -> Result<Self> {
        if active_mask == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "warp active_mask must have at least one lane active".into(),
            }
            .into());
        }
        Ok(Self { warp_size: DEFAULT_WARP_SIZE, active_mask })
    }

    /// Number of active lanes.
    pub fn active_count(&self) -> u32 {
        self.active_mask.count_ones()
    }

    /// Check if a lane is active.
    pub fn is_active(&self, lane: u32) -> bool {
        lane < self.warp_size && (self.active_mask & (1 << lane)) != 0
    }
}

// ---------------------------------------------------------------------------
// Warp-level reductions (CPU fallback — sequential simulation)
// ---------------------------------------------------------------------------

/// Butterfly-pattern warp-level sum reduction.
///
/// Simulates `__shfl_xor_sync`-based reduction. All active lanes receive
/// the sum of their values. Inactive lanes are left unchanged.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length does not
/// match `config.warp_size`.
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

/// Butterfly-pattern warp-level max reduction.
///
/// All active lanes receive the maximum value. Inactive lanes unchanged.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length mismatch.
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

/// Butterfly-pattern warp-level min reduction.
///
/// All active lanes receive the minimum value. Inactive lanes unchanged.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length mismatch.
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

// ---------------------------------------------------------------------------
// Warp-level communication
// ---------------------------------------------------------------------------

/// Broadcast a value from `src_lane` to all active lanes.
///
/// Simulates `__shfl_sync(mask, val, src_lane)`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `src_lane` is inactive
/// or out of range, or if `data` length mismatches.
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

/// Shuffle values between lanes using a source index per lane.
///
/// Simulates `__shfl_sync(mask, val, src_lane[i])` — each active lane `i`
/// reads from `src_lanes[i]`. Source lanes must be active.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes mismatch or
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
            if !config.is_active(src) {
                return Err(KernelError::InvalidArguments {
                    reason: format!("shuffle source lane {src} for dest lane {i} is not active"),
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

// ---------------------------------------------------------------------------
// Warp-level scans
// ---------------------------------------------------------------------------

/// Inclusive prefix sum within the warp.
///
/// After execution, `data[i]` contains the sum of `data[0..=i]` for all
/// active lanes. Inactive lanes keep original values.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length mismatch.
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

/// Exclusive prefix scan within the warp.
///
/// After execution, `data[i]` contains the sum of all active values
/// before lane `i`. The first active lane gets 0.0.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length mismatch.
pub fn warp_exclusive_scan(data: &mut [f32], config: &WarpConfig) -> Result<()> {
    validate_warp_data(data, config)?;
    let mut running = 0.0f32;
    for i in 0..config.warp_size {
        if config.is_active(i) {
            let val = data[i as usize];
            data[i as usize] = running;
            running += val;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Warp-level voting
// ---------------------------------------------------------------------------

/// Ballot vote across warp lanes.
///
/// Returns a bitmask where bit `i` is set iff lane `i` is active and
/// `predicates[i]` is true.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` length
/// does not match `config.warp_size`.
pub fn warp_ballot(predicates: &[bool], config: &WarpConfig) -> Result<u32> {
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
/// Equivalent to `__all_sync(mask, predicate)`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` length mismatch.
pub fn warp_all(predicates: &[bool], config: &WarpConfig) -> Result<bool> {
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
    Ok((0..config.warp_size).filter(|&i| config.is_active(i)).all(|i| predicates[i as usize]))
}

/// Check if any active lane satisfies the predicate.
///
/// Equivalent to `__any_sync(mask, predicate)`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` length mismatch.
pub fn warp_any(predicates: &[bool], config: &WarpConfig) -> Result<bool> {
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
    Ok((0..config.warp_size).filter(|&i| config.is_active(i)).any(|i| predicates[i as usize]))
}

/// Match lanes with the same value.
///
/// Returns a bitmask for each lane where bit `j` is set iff lane `j`
/// is active and holds the same value as lane `i`. Only active lanes
/// participate; inactive lane results are 0.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length mismatch.
pub fn warp_match(data: &[f32], config: &WarpConfig) -> Result<Vec<u32>> {
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
        let vi = data[i as usize];
        for j in 0..config.warp_size {
            if config.is_active(j) && data[j as usize] == vi {
                masks[i as usize] |= 1 << j;
            }
        }
    }
    Ok(masks)
}

// ---------------------------------------------------------------------------
// Block-level reductions
// ---------------------------------------------------------------------------

/// Block-level sum reduction using warp primitives.
///
/// Splits `data` into warps, reduces each warp, then reduces across warps.
/// Returns the total sum.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty.
pub fn block_reduce_sum(data: &[f32]) -> Result<f32> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "block_reduce_sum: data must not be empty".into(),
        }
        .into());
    }
    let ws = DEFAULT_WARP_SIZE as usize;
    let mut warp_sums = Vec::new();
    for chunk in data.chunks(ws) {
        let sum: f32 = chunk.iter().sum();
        warp_sums.push(sum);
    }
    // Second stage: reduce warp sums
    Ok(warp_sums.iter().sum())
}

/// Block-level max reduction using warp primitives.
///
/// Splits `data` into warps, finds max in each warp, then max across warps.
/// Returns the global maximum.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is empty.
pub fn block_reduce_max(data: &[f32]) -> Result<f32> {
    if data.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "block_reduce_max: data must not be empty".into(),
        }
        .into());
    }
    let ws = DEFAULT_WARP_SIZE as usize;
    let mut warp_maxes = Vec::new();
    for chunk in data.chunks(ws) {
        let max_val = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        warp_maxes.push(max_val);
    }
    Ok(warp_maxes.iter().copied().fold(f32::NEG_INFINITY, f32::max))
}

// ---------------------------------------------------------------------------
// Composite: cooperative softmax
// ---------------------------------------------------------------------------

/// Cooperative softmax using warp-level reductions.
///
/// Computes `softmax(input)` for each row of a `[n_rows, n_cols]` matrix
/// using the same algorithm as the CUDA kernel: per-row max reduction,
/// shifted exp + sum reduction, then normalization.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if dimensions are invalid or
/// buffer sizes are too small.
pub fn cooperative_softmax(
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) -> Result<()> {
    if n_rows == 0 || n_cols == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "cooperative_softmax: dimensions must be non-zero".into(),
        }
        .into());
    }
    let total = n_rows * n_cols;
    if input.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("cooperative_softmax: input length {} < {}", input.len(), total),
        }
        .into());
    }
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("cooperative_softmax: output length {} < {}", output.len(), total),
        }
        .into());
    }

    for row in 0..n_rows {
        let start = row * n_cols;
        let row_data = &input[start..start + n_cols];

        // Phase 1: row max (simulates warp reduction)
        let row_max = row_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Phase 2: shifted exp + sum (simulates warp reduction)
        let exp_sum: f32 = row_data.iter().map(|&v| (v - row_max).exp()).sum();

        // Phase 3: normalize
        let inv_sum = 1.0 / exp_sum;
        for c in 0..n_cols {
            output[start + c] = (input[start + c] - row_max).exp() * inv_sum;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

fn validate_warp_data(data: &[f32], config: &WarpConfig) -> Result<()> {
    if (data.len()) < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("warp data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // WarpConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_config_default() {
        let cfg = WarpConfig::default();
        assert_eq!(cfg.warp_size, 32);
        assert_eq!(cfg.active_mask, 0xFFFF_FFFF);
        assert_eq!(cfg.active_count(), 32);
    }

    #[test]
    fn test_warp_config_new() {
        let cfg = WarpConfig::new();
        assert_eq!(cfg, WarpConfig::default());
    }

    #[test]
    fn test_warp_config_with_mask() {
        let cfg = WarpConfig::with_mask(0x0000_00FF).unwrap();
        assert_eq!(cfg.active_count(), 8);
        assert!(cfg.is_active(0));
        assert!(cfg.is_active(7));
        assert!(!cfg.is_active(8));
    }

    #[test]
    fn test_warp_config_zero_mask_rejected() {
        assert!(WarpConfig::with_mask(0).is_err());
    }

    #[test]
    fn test_warp_config_is_active_out_of_range() {
        let cfg = WarpConfig::new();
        assert!(!cfg.is_active(32));
        assert!(!cfg.is_active(100));
    }

    #[test]
    fn test_warp_config_single_lane() {
        let cfg = WarpConfig::with_mask(1 << 15).unwrap();
        assert_eq!(cfg.active_count(), 1);
        assert!(!cfg.is_active(0));
        assert!(cfg.is_active(15));
    }

    // -----------------------------------------------------------------------
    // warp_reduce_sum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_all_active() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        for i in 0..32 {
            data[i] = (i + 1) as f32;
        }
        warp_reduce_sum(&mut data, &cfg).unwrap();
        let expected = (1..=32).sum::<u32>() as f32; // 528
        for &v in &data {
            assert!((v - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reduce_sum_partial_mask() {
        let cfg = WarpConfig::with_mask(0x0000_000F).unwrap(); // lanes 0–3
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[1] = 2.0;
        data[2] = 3.0;
        data[3] = 4.0;
        data[4] = 100.0; // inactive — must not change
        warp_reduce_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[3] - 10.0).abs() < 1e-5);
        assert!((data[4] - 100.0).abs() < 1e-5); // unchanged
    }

    #[test]
    fn test_reduce_sum_single_lane() {
        let cfg = WarpConfig::with_mask(1).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 42.0;
        warp_reduce_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_sum_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 16]; // too short
        assert!(warp_reduce_sum(&mut data, &cfg).is_err());
    }

    #[test]
    fn test_reduce_sum_zeros() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        warp_reduce_sum(&mut data, &cfg).unwrap();
        for &v in &data {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    fn test_reduce_sum_negative_values() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        for i in 0..32 {
            data[i] = -(i as f32);
        }
        warp_reduce_sum(&mut data, &cfg).unwrap();
        let expected: f32 = (0..32).map(|i| -(i as f32)).sum();
        assert!((data[0] - expected).abs() < 1e-3);
    }

    // -----------------------------------------------------------------------
    // warp_reduce_max tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_max_all_active() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        for i in 0..32 {
            data[i] = (i as f32) * 0.5;
        }
        warp_reduce_max(&mut data, &cfg).unwrap();
        let expected = 31.0 * 0.5;
        for &v in &data {
            assert!((v - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reduce_max_partial_mask() {
        let cfg = WarpConfig::with_mask(0b1010).unwrap(); // lanes 1, 3
        let mut data = vec![0.0f32; 32];
        data[0] = 999.0; // inactive
        data[1] = 3.0;
        data[3] = 7.0;
        warp_reduce_max(&mut data, &cfg).unwrap();
        assert!((data[1] - 7.0).abs() < 1e-5);
        assert!((data[3] - 7.0).abs() < 1e-5);
        assert!((data[0] - 999.0).abs() < 1e-5); // unchanged
    }

    #[test]
    fn test_reduce_max_negative_values() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| -100.0 + i as f32).collect();
        warp_reduce_max(&mut data, &cfg).unwrap();
        assert!((data[0] - (-69.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_max_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 10];
        assert!(warp_reduce_max(&mut data, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_reduce_min tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_min_all_active() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| (i + 10) as f32).collect();
        warp_reduce_min(&mut data, &cfg).unwrap();
        for &v in &data {
            assert!((v - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reduce_min_partial_mask() {
        let cfg = WarpConfig::with_mask(0b1100).unwrap(); // lanes 2, 3
        let mut data = vec![0.0f32; 32];
        data[0] = -999.0; // inactive
        data[2] = 5.0;
        data[3] = 2.0;
        warp_reduce_min(&mut data, &cfg).unwrap();
        assert!((data[2] - 2.0).abs() < 1e-5);
        assert!((data[3] - 2.0).abs() < 1e-5);
        assert!((data[0] - (-999.0)).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_min_single_element() {
        let cfg = WarpConfig::with_mask(1 << 5).unwrap();
        let mut data = vec![0.0f32; 32];
        data[5] = 77.0;
        warp_reduce_min(&mut data, &cfg).unwrap();
        assert!((data[5] - 77.0).abs() < 1e-5);
    }

    #[test]
    fn test_reduce_min_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 4];
        assert!(warp_reduce_min(&mut data, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_broadcast tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_broadcast_from_lane_zero() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        data[0] = 42.0;
        warp_broadcast(&mut data, 0, &cfg).unwrap();
        for &v in &data {
            assert!((v - 42.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_broadcast_from_last_lane() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        warp_broadcast(&mut data, 31, &cfg).unwrap();
        for &v in &data {
            assert!((v - 31.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_broadcast_partial_mask() {
        let cfg = WarpConfig::with_mask(0b1111).unwrap(); // lanes 0–3
        let mut data = vec![0.0f32; 32];
        data[2] = 10.0;
        data[10] = 99.0; // inactive, should stay
        warp_broadcast(&mut data, 2, &cfg).unwrap();
        assert!((data[0] - 10.0).abs() < 1e-7);
        assert!((data[3] - 10.0).abs() < 1e-7);
        assert!((data[10] - 99.0).abs() < 1e-7);
    }

    #[test]
    fn test_broadcast_inactive_source_rejected() {
        let cfg = WarpConfig::with_mask(0b0001).unwrap(); // only lane 0
        let mut data = vec![0.0f32; 32];
        assert!(warp_broadcast(&mut data, 1, &cfg).is_err());
    }

    #[test]
    fn test_broadcast_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 8];
        assert!(warp_broadcast(&mut data, 0, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_shuffle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shuffle_identity() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src: Vec<u32> = (0..32).collect();
        let original = data.clone();
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn test_shuffle_reverse() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src: Vec<u32> = (0..32).rev().collect();
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - (31 - i) as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn test_shuffle_broadcast_via_shuffle() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src: Vec<u32> = vec![5; 32]; // all read from lane 5
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        for &v in &data {
            assert!((v - 5.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_shuffle_src_lanes_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        let src = vec![0u32; 16]; // too short
        assert!(warp_shuffle(&mut data, &src, &cfg).is_err());
    }

    #[test]
    fn test_shuffle_inactive_source_rejected() {
        let cfg = WarpConfig::with_mask(0b0011).unwrap(); // lanes 0, 1
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[1] = 2.0;
        // Lane 0 tries to read from lane 5 (inactive) → error
        let mut src: Vec<u32> = (0..32).collect();
        src[0] = 5;
        assert!(warp_shuffle(&mut data, &src, &cfg).is_err());
    }

    #[test]
    fn test_shuffle_swap_pairs() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let src: Vec<u32> = (0..32).map(|i| i ^ 1).collect(); // XOR swap
        warp_shuffle(&mut data, &src, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
        assert!((data[1] - 0.0).abs() < 1e-7);
        assert!((data[2] - 3.0).abs() < 1e-7);
        assert!((data[3] - 2.0).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // warp_prefix_sum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_sum_sequential() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 32];
        warp_prefix_sum(&mut data, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - (i + 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_prefix_sum_varying() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        warp_prefix_sum(&mut data, &cfg).unwrap();
        // data[i] = 1 + 2 + ... + (i+1) = (i+1)(i+2)/2
        for i in 0..32 {
            let expected = ((i + 1) * (i + 2)) as f32 / 2.0;
            assert!((data[i] - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn test_prefix_sum_partial_mask() {
        // Only lanes 0, 2, 4 active
        let cfg = WarpConfig::with_mask(0b10101).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[1] = 999.0; // inactive
        data[2] = 2.0;
        data[4] = 3.0;
        warp_prefix_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 999.0).abs() < 1e-5); // unchanged
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[4] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_prefix_sum_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 2];
        assert!(warp_prefix_sum(&mut data, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_exclusive_scan tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exclusive_scan_ones() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 32];
        warp_exclusive_scan(&mut data, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_exclusive_scan_first_is_zero() {
        let cfg = WarpConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        warp_exclusive_scan(&mut data, &cfg).unwrap();
        assert!(data[0].abs() < 1e-7);
    }

    #[test]
    fn test_exclusive_scan_partial_mask() {
        let cfg = WarpConfig::with_mask(0b111).unwrap(); // lanes 0, 1, 2
        let mut data = vec![0.0f32; 32];
        data[0] = 10.0;
        data[1] = 20.0;
        data[2] = 30.0;
        data[5] = 999.0; // inactive
        warp_exclusive_scan(&mut data, &cfg).unwrap();
        assert!(data[0].abs() < 1e-7); // first active → 0
        assert!((data[1] - 10.0).abs() < 1e-5);
        assert!((data[2] - 30.0).abs() < 1e-5);
        assert!((data[5] - 999.0).abs() < 1e-5); // unchanged
    }

    #[test]
    fn test_exclusive_scan_data_too_short() {
        let cfg = WarpConfig::new();
        let mut data = vec![1.0f32; 3];
        assert!(warp_exclusive_scan(&mut data, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_ballot tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ballot_all_true() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        let result = warp_ballot(&preds, &cfg).unwrap();
        assert_eq!(result, 0xFFFF_FFFF);
    }

    #[test]
    fn test_ballot_all_false() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 32];
        let result = warp_ballot(&preds, &cfg).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_ballot_alternating() {
        let cfg = WarpConfig::new();
        let preds: Vec<bool> = (0..32).map(|i| i % 2 == 0).collect();
        let result = warp_ballot(&preds, &cfg).unwrap();
        assert_eq!(result, 0x5555_5555);
    }

    #[test]
    fn test_ballot_partial_mask() {
        let cfg = WarpConfig::with_mask(0x0000_00FF).unwrap(); // lanes 0–7
        let preds = vec![true; 32];
        let result = warp_ballot(&preds, &cfg).unwrap();
        assert_eq!(result, 0x0000_00FF);
    }

    #[test]
    fn test_ballot_predicates_too_short() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 16];
        assert!(warp_ballot(&preds, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_all / warp_any tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_warp_all_true() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        assert!(warp_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_warp_all_one_false() {
        let cfg = WarpConfig::new();
        let mut preds = vec![true; 32];
        preds[15] = false;
        assert!(!warp_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_warp_all_partial_mask_ignores_inactive() {
        let cfg = WarpConfig::with_mask(0b11).unwrap(); // lanes 0, 1
        let mut preds = vec![false; 32]; // inactive lanes false
        preds[0] = true;
        preds[1] = true;
        assert!(warp_all(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_warp_any_all_false() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 32];
        assert!(!warp_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_warp_any_one_true() {
        let cfg = WarpConfig::new();
        let mut preds = vec![false; 32];
        preds[31] = true;
        assert!(warp_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_warp_any_partial_mask_ignores_inactive() {
        let cfg = WarpConfig::with_mask(0b01).unwrap(); // only lane 0
        let mut preds = vec![false; 32];
        preds[1] = true; // inactive lane — not counted
        assert!(!warp_any(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_warp_all_predicates_too_short() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 10];
        assert!(warp_all(&preds, &cfg).is_err());
    }

    #[test]
    fn test_warp_any_predicates_too_short() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 10];
        assert!(warp_any(&preds, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // warp_match tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_match_all_same() {
        let cfg = WarpConfig::new();
        let data = vec![7.0f32; 32];
        let masks = warp_match(&data, &cfg).unwrap();
        for &m in &masks {
            assert_eq!(m, 0xFFFF_FFFF);
        }
    }

    #[test]
    fn test_match_all_different() {
        let cfg = WarpConfig::new();
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let masks = warp_match(&data, &cfg).unwrap();
        for i in 0..32 {
            assert_eq!(masks[i], 1 << i);
        }
    }

    #[test]
    fn test_match_groups() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        // Group A: lanes 0–7 = 1.0, Group B: lanes 8–15 = 2.0, rest = 3.0
        for i in 0..8 {
            data[i] = 1.0;
        }
        for i in 8..16 {
            data[i] = 2.0;
        }
        for i in 16..32 {
            data[i] = 3.0;
        }
        let masks = warp_match(&data, &cfg).unwrap();
        assert_eq!(masks[0], 0x0000_00FF); // lanes 0–7
        assert_eq!(masks[8], 0x0000_FF00); // lanes 8–15
        assert_eq!(masks[16], 0xFFFF_0000); // lanes 16–31
    }

    #[test]
    fn test_match_partial_mask() {
        let cfg = WarpConfig::with_mask(0b1111).unwrap();
        let data = vec![5.0f32; 32];
        let masks = warp_match(&data, &cfg).unwrap();
        assert_eq!(masks[0], 0b1111); // only active lanes match
        assert_eq!(masks[4], 0); // inactive lane → 0
    }

    #[test]
    fn test_match_data_too_short() {
        let cfg = WarpConfig::new();
        let data = vec![0.0f32; 10];
        assert!(warp_match(&data, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // block_reduce_sum tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_reduce_sum_single_warp() {
        let data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let sum = block_reduce_sum(&data).unwrap();
        assert!((sum - 528.0).abs() < 1e-3);
    }

    #[test]
    fn test_block_reduce_sum_multiple_warps() {
        let data = vec![1.0f32; 128]; // 4 warps
        let sum = block_reduce_sum(&data).unwrap();
        assert!((sum - 128.0).abs() < 1e-3);
    }

    #[test]
    fn test_block_reduce_sum_partial_last_warp() {
        let data = vec![1.0f32; 50]; // 1 full warp + 18 leftover
        let sum = block_reduce_sum(&data).unwrap();
        assert!((sum - 50.0).abs() < 1e-3);
    }

    #[test]
    fn test_block_reduce_sum_empty() {
        let data: Vec<f32> = vec![];
        assert!(block_reduce_sum(&data).is_err());
    }

    #[test]
    fn test_block_reduce_sum_single_element() {
        let data = vec![42.0f32];
        let sum = block_reduce_sum(&data).unwrap();
        assert!((sum - 42.0).abs() < 1e-7);
    }

    #[test]
    fn test_block_reduce_sum_large_block() {
        let data = vec![0.5f32; 1024]; // 32 warps
        let sum = block_reduce_sum(&data).unwrap();
        assert!((sum - 512.0).abs() < 1e-1);
    }

    // -----------------------------------------------------------------------
    // block_reduce_max tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_reduce_max_single_warp() {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let max = block_reduce_max(&data).unwrap();
        assert!((max - 31.0).abs() < 1e-5);
    }

    #[test]
    fn test_block_reduce_max_multiple_warps() {
        let mut data = vec![0.0f32; 128];
        data[65] = 100.0;
        let max = block_reduce_max(&data).unwrap();
        assert!((max - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_block_reduce_max_negative() {
        let data: Vec<f32> = (0..64).map(|i| -100.0 + i as f32).collect();
        let max = block_reduce_max(&data).unwrap();
        assert!((max - (-37.0)).abs() < 1e-5);
    }

    #[test]
    fn test_block_reduce_max_empty() {
        let data: Vec<f32> = vec![];
        assert!(block_reduce_max(&data).is_err());
    }

    #[test]
    fn test_block_reduce_max_single_element() {
        let data = vec![-5.0f32];
        let max = block_reduce_max(&data).unwrap();
        assert!((max - (-5.0)).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // cooperative_softmax tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_single_row() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        cooperative_softmax(&input, &mut output, 1, 4).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Values should be monotonically increasing
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
        assert!(output[2] < output[3]);
    }

    #[test]
    fn test_softmax_uniform() {
        let input = vec![1.0f32; 8];
        let mut output = vec![0.0f32; 8];
        cooperative_softmax(&input, &mut output, 1, 8).unwrap();
        for &v in &output {
            assert!((v - 0.125).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_multiple_rows() {
        let input = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; 6];
        cooperative_softmax(&input, &mut output, 2, 3).unwrap();
        // Each row sums to 1
        let sum1: f32 = output[0..3].iter().sum();
        let sum2: f32 = output[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
        // Second row (all zeros) should be uniform
        for &v in &output[3..6] {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow naive exp
        let input = vec![1000.0, 1001.0, 1002.0, 999.0];
        let mut output = vec![0.0f32; 4];
        cooperative_softmax(&input, &mut output, 1, 4).unwrap();
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_softmax_zero_rows_rejected() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        assert!(cooperative_softmax(&input, &mut output, 0, 4).is_err());
    }

    #[test]
    fn test_softmax_zero_cols_rejected() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 4];
        assert!(cooperative_softmax(&input, &mut output, 4, 0).is_err());
    }

    #[test]
    fn test_softmax_input_too_short() {
        let input = vec![1.0f32; 3];
        let mut output = vec![0.0f32; 8];
        assert!(cooperative_softmax(&input, &mut output, 2, 4).is_err());
    }

    #[test]
    fn test_softmax_output_too_short() {
        let input = vec![1.0f32; 8];
        let mut output = vec![0.0f32; 3];
        assert!(cooperative_softmax(&input, &mut output, 2, 4).is_err());
    }

    #[test]
    fn test_softmax_peaked_distribution() {
        // One very large value should dominate
        let mut input = vec![0.0f32; 32];
        input[10] = 50.0;
        let mut output = vec![0.0f32; 32];
        cooperative_softmax(&input, &mut output, 1, 32).unwrap();
        assert!(output[10] > 0.99);
    }

    // -----------------------------------------------------------------------
    // Consistency and integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_matches_block_reduce() {
        let data: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let block_sum = block_reduce_sum(&data).unwrap();
        let cfg = WarpConfig::new();
        let mut warp_data = data.clone();
        warp_reduce_sum(&mut warp_data, &cfg).unwrap();
        assert!((warp_data[0] - block_sum).abs() < 1e-3);
    }

    #[test]
    fn test_reduce_max_matches_block_reduce() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.3).collect();
        let block_max = block_reduce_max(&data).unwrap();
        let cfg = WarpConfig::new();
        let mut warp_data = data.clone();
        warp_reduce_max(&mut warp_data, &cfg).unwrap();
        assert!((warp_data[0] - block_max).abs() < 1e-5);
    }

    #[test]
    fn test_prefix_sum_last_equals_reduce_sum() {
        let cfg = WarpConfig::new();
        let values: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let mut prefix = values.clone();
        warp_prefix_sum(&mut prefix, &cfg).unwrap();
        let mut reduced = values;
        warp_reduce_sum(&mut reduced, &cfg).unwrap();
        assert!((prefix[31] - reduced[0]).abs() < 1e-3);
    }

    #[test]
    fn test_exclusive_scan_shift_of_inclusive() {
        let cfg = WarpConfig::new();
        let values = vec![2.0f32; 32];
        let mut inclusive = values.clone();
        warp_prefix_sum(&mut inclusive, &cfg).unwrap();
        let mut exclusive = values;
        warp_exclusive_scan(&mut exclusive, &cfg).unwrap();
        // exclusive[i] = inclusive[i] - original[i]
        for i in 0..32 {
            assert!((exclusive[i] - (inclusive[i] - 2.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ballot_matches_all_any() {
        let cfg = WarpConfig::new();
        let preds = vec![true; 32];
        let ballot = warp_ballot(&preds, &cfg).unwrap();
        let all = warp_all(&preds, &cfg).unwrap();
        let any = warp_any(&preds, &cfg).unwrap();
        assert_eq!(ballot, 0xFFFF_FFFF);
        assert!(all);
        assert!(any);
    }

    #[test]
    fn test_ballot_empty_matches_all_any() {
        let cfg = WarpConfig::new();
        let preds = vec![false; 32];
        let ballot = warp_ballot(&preds, &cfg).unwrap();
        let all = warp_all(&preds, &cfg).unwrap();
        let any = warp_any(&preds, &cfg).unwrap();
        assert_eq!(ballot, 0);
        assert!(!all);
        assert!(!any);
    }

    #[test]
    fn test_broadcast_is_idempotent() {
        let cfg = WarpConfig::new();
        let mut data = vec![0.0f32; 32];
        data[0] = 5.0;
        warp_broadcast(&mut data, 0, &cfg).unwrap();
        let after_first = data.clone();
        warp_broadcast(&mut data, 0, &cfg).unwrap();
        assert_eq!(data, after_first);
    }

    #[test]
    fn test_shuffle_then_reduce_commutes() {
        let cfg = WarpConfig::new();
        let original: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();

        // reduce original
        let mut reduced = original.clone();
        warp_reduce_sum(&mut reduced, &cfg).unwrap();

        // shuffle (reverse) then reduce — sum should be the same
        let src: Vec<u32> = (0..32).rev().collect();
        let mut shuffled = original;
        warp_shuffle(&mut shuffled, &src, &cfg).unwrap();
        warp_reduce_sum(&mut shuffled, &cfg).unwrap();

        assert!((reduced[0] - shuffled[0]).abs() < 1e-3);
    }

    #[test]
    fn test_match_symmetry() {
        let cfg = WarpConfig::new();
        let data: Vec<f32> = (0..32).map(|i| (i % 4) as f32).collect();
        let masks = warp_match(&data, &cfg).unwrap();
        // If lane i matches lane j, then lane j matches lane i
        for i in 0..32 {
            for j in 0..32 {
                let i_matches_j = (masks[i] >> j) & 1 == 1;
                let j_matches_i = (masks[j] >> i) & 1 == 1;
                assert_eq!(i_matches_j, j_matches_i);
            }
        }
    }
}
