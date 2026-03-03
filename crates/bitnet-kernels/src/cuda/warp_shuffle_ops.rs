//! Warp shuffle operations for CUDA-style parallel computation.
//!
//! This module provides CPU-simulated warp shuffle primitives that mirror
//! CUDA's `__shfl_sync`, `__shfl_xor_sync`, `__shfl_down_sync`, and
//! `__shfl_up_sync` intrinsics, along with higher-level patterns built
//! on top of them:
//!
//! # Shuffle primitives
//!
//! - [`shuffle_xor`] — XOR-indexed lane shuffle (`__shfl_xor_sync`)
//! - [`shuffle_down`] — downward delta shuffle (`__shfl_down_sync`)
//! - [`shuffle_up`] — upward delta shuffle (`__shfl_up_sync`)
//! - [`shuffle_idx`] — direct indexed shuffle (`__shfl_sync`)
//!
//! # Butterfly reductions
//!
//! - [`butterfly_reduce_sum`] — sum via iterative XOR shuffles
//! - [`butterfly_reduce_max`] — max via iterative XOR shuffles
//! - [`butterfly_reduce_min`] — min via iterative XOR shuffles
//!
//! # Halving reductions
//!
//! - [`halving_reduce_sum`] — sum via `shuffle_down` halving pattern
//! - [`halving_reduce_max`] — max via `shuffle_down` halving pattern
//!
//! # Scans
//!
//! - [`shuffle_inclusive_scan`] — inclusive prefix sum via shuffle-up
//! - [`shuffle_exclusive_scan`] — exclusive prefix sum via shuffle-up
//! - [`segmented_inclusive_scan`] — prefix sum with segment boundaries
//!
//! # Cross-warp patterns
//!
//! - [`butterfly_exchange`] — butterfly data exchange at a given stage
//! - [`cross_warp_reduce_sum`] — multi-warp sum reduction
//! - [`cross_warp_reduce_max`] — multi-warp max reduction
//!
//! # Warp-synchronized collectives
//!
//! - [`warp_allgather`] — gather all lane values into every lane
//! - [`warp_scatter`] — distribute from one lane to specified targets
//! - [`warp_sync_reduce_sum`] — synchronized reduction with barrier flag
//!
//! # Vote / divergence
//!
//! - [`divergence_ballot`] — bitmask of lanes with divergent values
//! - [`uniform_branch_check`] — test if all active lanes agree
//! - [`active_lane_count`] — popcount of active mask
//! - [`first_active_lane`] — index of first active lane
//! - [`popc`] — population count of a ballot mask
//!
//! # Matrix fragment operations (tensor core simulation)
//!
//! - [`MatrixFragment`] — small matrix fragment held across warp lanes
//! - [`fragment_load`] — load fragment from buffer
//! - [`fragment_store`] — store fragment to buffer
//! - [`fragment_fill`] — fill fragment with a scalar
//! - [`fragment_mma`] — multiply-accumulate: D = A·B + C
//!
//! # Shuffle-based transpose
//!
//! - [`shuffle_transpose_4x8`] — transpose a 4×8 sub-matrix within a warp
//! - [`shuffle_transpose_8x4`] — transpose an 8×4 sub-matrix within a warp
//!
//! # Warp specialization
//!
//! - [`WarpRole`] — producer / consumer role tag
//! - [`WarpSpecConfig`] — split a warp into producer and consumer halves
//! - [`warp_specialize_map`] — apply different functions per role
//! - [`warp_pipeline_stages`] — multi-stage pipeline within a warp
//!
//! # CUDA kernel source
//!
//! [`WARP_SHUFFLE_OPS_KERNEL_SRC`] contains CUDA C kernels that use hardware
//! shuffle intrinsics. Feature-gated behind
//! `#[cfg(any(feature = "gpu", feature = "cuda"))]`.

use bitnet_common::{KernelError, Result};

/// Default CUDA warp size.
pub const WARP_SIZE: u32 = 32;

// =========================================================================
// CUDA kernel source
// =========================================================================

/// CUDA C kernel source implementing warp shuffle operations.
///
/// Contains kernels for butterfly reductions, halving reductions,
/// inclusive/exclusive scans, matrix-fragment MMA, and shuffle-based
/// transpose using `__shfl_xor_sync`, `__shfl_down_sync`, and
/// `__shfl_up_sync` intrinsics.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const WARP_SHUFFLE_OPS_KERNEL_SRC: &str = r#"
// Butterfly sum reduction via __shfl_xor_sync.
extern "C" __global__ void butterfly_reduce_sum_f32(
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

// Halving sum reduction via __shfl_down_sync.
extern "C" __global__ void halving_reduce_sum_f32(
    float* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    const unsigned MASK = 0xFFFFFFFFu;
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_down_sync(MASK, val, offset);
    }
    data[idx] = val;
}

// Inclusive prefix sum via __shfl_up_sync (Hillis-Steele).
extern "C" __global__ void shuffle_inclusive_scan_f32(
    float* __restrict__ data,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = data[idx];
    int lane = threadIdx.x & 31;
    const unsigned MASK = 0xFFFFFFFFu;
    for (int d = 1; d < 32; d <<= 1) {
        float tmp = __shfl_up_sync(MASK, val, d);
        if (lane >= d) val += tmp;
    }
    data[idx] = val;
}

// Shuffle-based 4x8 transpose within a warp.
// Each lane holds one element; 32 lanes = 4 rows x 8 cols.
// After: lane (r*8+c) holds input at (c*4+r) -> transposed to 8x4.
extern "C" __global__ void shuffle_transpose_4x8_f32(
    float* __restrict__ data,
    int n_blocks)
{
    int block = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    if (block >= n_blocks) return;
    int lane = threadIdx.x & 31;
    int base = block * 32;
    float val = data[base + lane];
    int row = lane / 8;   // 0..3
    int col = lane % 8;   // 0..7
    int src = col * 4 + row;
    val = __shfl_sync(0xFFFFFFFFu, val, src);
    data[base + lane] = val;
}

// Fragment MMA: D = A * B + C  (4x4 tiles, f32).
// A is row-major 4x4, B is col-major 4x4, C and D are row-major 4x4.
// Lanes 0..15 each compute one element of D.
extern "C" __global__ void fragment_mma_4x4_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float*       __restrict__ D,
    int n_frags)
{
    int frag = blockIdx.x * blockDim.x / 16 + threadIdx.x / 16;
    if (frag >= n_frags) return;
    int lane = threadIdx.x & 15;
    int r = lane / 4;
    int c = lane % 4;
    int off = frag * 16;
    float acc = C[off + lane];
    for (int k = 0; k < 4; k++) {
        float a_val = __shfl_sync(0xFFFFu, A[off + r * 4 + k], r * 4 + k);
        float b_val = __shfl_sync(0xFFFFu, B[off + c * 4 + k], c * 4 + k);
        acc += a_val * b_val;
    }
    D[off + lane] = acc;
}
"#;

// =========================================================================
// Configuration
// =========================================================================

/// Configuration for warp shuffle operations.
///
/// Mirrors the active-mask concept of CUDA warp intrinsics. A full mask
/// (`0xFFFF_FFFF`) means all 32 lanes participate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShuffleConfig {
    /// Number of lanes in the warp (always 32 for real CUDA hardware).
    pub warp_size: u32,
    /// Bitmask of active lanes.
    pub active_mask: u32,
}

impl Default for ShuffleConfig {
    fn default() -> Self {
        Self { warp_size: WARP_SIZE, active_mask: 0xFFFF_FFFF }
    }
}

impl ShuffleConfig {
    /// Full warp, all lanes active.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a custom active mask. At least one lane must be active.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `mask` is zero.
    pub fn with_mask(mask: u32) -> Result<Self> {
        if mask == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "active mask must not be zero".into(),
            }
            .into());
        }
        Ok(Self { warp_size: WARP_SIZE, active_mask: mask })
    }

    /// True if lane `i` is active.
    #[inline]
    #[must_use]
    pub fn is_active(&self, lane: u32) -> bool {
        lane < self.warp_size && (self.active_mask >> lane) & 1 == 1
    }

    /// Number of active lanes.
    #[must_use]
    pub fn active_count(&self) -> u32 {
        self.active_mask.count_ones()
    }
}

// =========================================================================
// Validation helper
// =========================================================================

fn validate_data(data: &[f32], config: &ShuffleConfig) -> Result<()> {
    if data.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    Ok(())
}

// =========================================================================
// Shuffle primitives
// =========================================================================

/// XOR-indexed lane shuffle.
///
/// Each active lane `i` receives the value from lane `i ^ xor_mask`.
/// The source lane must be active; if it is not, the lane keeps its
/// original value (matching CUDA shfl_xor_sync semantics for inactive
/// source lanes).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn shuffle_xor(data: &mut [f32], xor_mask: u32, config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
    for i in 0..config.warp_size {
        if config.is_active(i) {
            let src = i ^ xor_mask;
            if src < config.warp_size && config.is_active(src) {
                data[i as usize] = snapshot[src as usize];
            }
        }
    }
    Ok(())
}

/// Downward delta shuffle.
///
/// Each active lane `i` receives the value from lane `i + delta`.
/// If the source lane is out of range or inactive the lane keeps its
/// original value.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn shuffle_down(data: &mut [f32], delta: u32, config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
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

/// Upward delta shuffle.
///
/// Each active lane `i` receives the value from lane `i - delta`.
/// If the source lane is out of range or inactive the lane keeps its
/// original value.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn shuffle_up(data: &mut [f32], delta: u32, config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
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

/// Direct indexed shuffle.
///
/// Each active lane `i` receives the value from lane `src_lane`.
/// Equivalent to broadcast when all lanes use the same `src_lane`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short or
/// `src_lane` is out of range.
pub fn shuffle_idx(data: &mut [f32], src_lane: u32, config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    if src_lane >= config.warp_size {
        return Err(KernelError::InvalidArguments {
            reason: format!("src_lane {} >= warp_size {}", src_lane, config.warp_size),
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

// =========================================================================
// Butterfly reductions (via XOR shuffles)
// =========================================================================

/// Butterfly sum reduction via iterative XOR shuffles.
///
/// After execution every active lane holds the sum of all active lane
/// values. This mirrors the CUDA pattern:
/// ```text
/// for offset in [16, 8, 4, 2, 1]:
///     val += __shfl_xor_sync(MASK, val, offset)
/// ```
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn butterfly_reduce_sum(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let mut offset = config.warp_size >> 1;
    while offset >= 1 {
        let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
        for i in 0..config.warp_size {
            if config.is_active(i) {
                let src = i ^ offset;
                if src < config.warp_size && config.is_active(src) {
                    data[i as usize] += snapshot[src as usize];
                }
            }
        }
        offset >>= 1;
    }
    Ok(())
}

/// Butterfly max reduction via iterative XOR shuffles.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn butterfly_reduce_max(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let mut offset = config.warp_size >> 1;
    while offset >= 1 {
        let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
        for i in 0..config.warp_size {
            if config.is_active(i) {
                let src = i ^ offset;
                if src < config.warp_size && config.is_active(src) {
                    data[i as usize] = data[i as usize].max(snapshot[src as usize]);
                }
            }
        }
        offset >>= 1;
    }
    Ok(())
}

/// Butterfly min reduction via iterative XOR shuffles.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn butterfly_reduce_min(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let mut offset = config.warp_size >> 1;
    while offset >= 1 {
        let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
        for i in 0..config.warp_size {
            if config.is_active(i) {
                let src = i ^ offset;
                if src < config.warp_size && config.is_active(src) {
                    data[i as usize] = data[i as usize].min(snapshot[src as usize]);
                }
            }
        }
        offset >>= 1;
    }
    Ok(())
}

// =========================================================================
// Halving reductions (via shuffle-down)
// =========================================================================

/// Halving sum reduction via shuffle-down.
///
/// After execution lane 0 holds the total sum. Higher lanes hold
/// partial results (matching CUDA `__shfl_down_sync` reduction).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn halving_reduce_sum(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let mut offset = config.warp_size >> 1;
    while offset >= 1 {
        let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
        for i in 0..config.warp_size {
            if config.is_active(i) {
                let src = i + offset;
                if src < config.warp_size && config.is_active(src) {
                    data[i as usize] += snapshot[src as usize];
                }
            }
        }
        offset >>= 1;
    }
    Ok(())
}

/// Halving max reduction via shuffle-down.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn halving_reduce_max(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let mut offset = config.warp_size >> 1;
    while offset >= 1 {
        let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
        for i in 0..config.warp_size {
            if config.is_active(i) {
                let src = i + offset;
                if src < config.warp_size && config.is_active(src) {
                    data[i as usize] = data[i as usize].max(snapshot[src as usize]);
                }
            }
        }
        offset >>= 1;
    }
    Ok(())
}

// =========================================================================
// Shuffle-based scans
// =========================================================================

/// Inclusive prefix sum via shuffle-up (Hillis–Steele pattern).
///
/// After execution, `data[i]` holds the sum of `data[0..=i]` for
/// active lanes. Mirrors the CUDA pattern:
/// ```text
/// for d in [1, 2, 4, 8, 16]:
///     tmp = __shfl_up_sync(MASK, val, d)
///     if lane >= d: val += tmp
/// ```
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn shuffle_inclusive_scan(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let mut d = 1u32;
    while d < config.warp_size {
        let snapshot: Vec<f32> = data[..config.warp_size as usize].to_vec();
        for i in 0..config.warp_size {
            if config.is_active(i) && i >= d {
                let src = i - d;
                if config.is_active(src) {
                    data[i as usize] += snapshot[src as usize];
                }
            }
        }
        d <<= 1;
    }
    Ok(())
}

/// Exclusive prefix sum via shuffle-up.
///
/// After execution, `data[i]` holds the sum of `data[0..i]` for
/// active lanes. The first active lane gets 0.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn shuffle_exclusive_scan(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    // Save originals, compute inclusive, then shift.
    let originals: Vec<f32> = data[..config.warp_size as usize].to_vec();
    shuffle_inclusive_scan(data, config)?;
    for i in 0..config.warp_size {
        if config.is_active(i) {
            data[i as usize] -= originals[i as usize];
        }
    }
    Ok(())
}

/// Segmented inclusive prefix sum.
///
/// `segment_heads[i]` is `true` if lane `i` starts a new segment.
/// The prefix sum resets at each segment boundary.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes mismatch.
pub fn segmented_inclusive_scan(
    data: &mut [f32],
    segment_heads: &[bool],
    config: &ShuffleConfig,
) -> Result<()> {
    validate_data(data, config)?;
    if segment_heads.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "segment_heads length {} < warp_size {}",
                segment_heads.len(),
                config.warp_size
            ),
        }
        .into());
    }
    let mut running = 0.0f32;
    for i in 0..config.warp_size {
        if !config.is_active(i) {
            continue;
        }
        if segment_heads[i as usize] {
            running = 0.0;
        }
        running += data[i as usize];
        data[i as usize] = running;
    }
    Ok(())
}

// =========================================================================
// Cross-warp communication patterns
// =========================================================================

/// Butterfly data exchange at a specific stage.
///
/// At stage `s`, each lane `i` exchanges its value with lane `i ^ (1 << s)`.
/// This is one step of the butterfly network used in FFT-like algorithms.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short or
/// `stage` is out of range.
pub fn butterfly_exchange(data: &mut [f32], stage: u32, config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let xor_mask = 1u32 << stage;
    if xor_mask >= config.warp_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "butterfly stage {} produces mask {} >= warp_size {}",
                stage, xor_mask, config.warp_size
            ),
        }
        .into());
    }
    shuffle_xor(data, xor_mask, config)
}

/// Cross-warp sum reduction.
///
/// Reduces values from multiple warps (each represented as a slice of
/// `warp_size` elements) into a single warp-sized output buffer.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any warp slice is too
/// short or `output` is too short.
pub fn cross_warp_reduce_sum(
    warps: &[&[f32]],
    output: &mut [f32],
    config: &ShuffleConfig,
) -> Result<()> {
    let ws = config.warp_size as usize;
    if output.len() < ws {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < warp_size {}", output.len(), config.warp_size),
        }
        .into());
    }
    for (idx, warp) in warps.iter().enumerate() {
        if warp.len() < ws {
            return Err(KernelError::InvalidArguments {
                reason: format!("warp[{idx}] length {} < warp_size {}", warp.len(), ws),
            }
            .into());
        }
    }
    for lane in 0..ws {
        if config.is_active(lane as u32) {
            output[lane] = warps.iter().map(|w| w[lane]).sum();
        } else {
            output[lane] = 0.0;
        }
    }
    Ok(())
}

/// Cross-warp max reduction.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if any warp slice or `output`
/// is too short.
pub fn cross_warp_reduce_max(
    warps: &[&[f32]],
    output: &mut [f32],
    config: &ShuffleConfig,
) -> Result<()> {
    let ws = config.warp_size as usize;
    if output.len() < ws {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < warp_size {}", output.len(), config.warp_size),
        }
        .into());
    }
    if warps.is_empty() {
        return Err(KernelError::InvalidArguments {
            reason: "cross_warp_reduce_max: need at least one warp".into(),
        }
        .into());
    }
    for (idx, warp) in warps.iter().enumerate() {
        if warp.len() < ws {
            return Err(KernelError::InvalidArguments {
                reason: format!("warp[{idx}] length {} < warp_size {}", warp.len(), ws),
            }
            .into());
        }
    }
    for lane in 0..ws {
        if config.is_active(lane as u32) {
            output[lane] = warps.iter().map(|w| w[lane]).fold(f32::NEG_INFINITY, f32::max);
        } else {
            output[lane] = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

// =========================================================================
// Warp-synchronized collectives
// =========================================================================

/// Gather all lane values so every lane sees the complete array.
///
/// Returns a `Vec<f32>` of length `warp_size` where index `j` is the
/// value from lane `j`. Inactive lanes contribute 0.0.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn warp_allgather(data: &[f32], config: &ShuffleConfig) -> Result<Vec<f32>> {
    if data.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    let ws = config.warp_size as usize;
    let mut gathered = vec![0.0f32; ws];
    for i in 0..ws {
        if config.is_active(i as u32) {
            gathered[i] = data[i];
        }
    }
    Ok(gathered)
}

/// Scatter a value from `src_lane` to specified target lanes.
///
/// `targets` is a bitmask of destination lanes. Only active target
/// lanes are written.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short or
/// `src_lane` is out of range.
pub fn warp_scatter(
    data: &mut [f32],
    src_lane: u32,
    targets: u32,
    config: &ShuffleConfig,
) -> Result<()> {
    validate_data(data, config)?;
    if src_lane >= config.warp_size {
        return Err(KernelError::InvalidArguments {
            reason: format!("src_lane {} >= warp_size {}", src_lane, config.warp_size),
        }
        .into());
    }
    let val = data[src_lane as usize];
    for i in 0..config.warp_size {
        if config.is_active(i) && (targets >> i) & 1 == 1 {
            data[i as usize] = val;
        }
    }
    Ok(())
}

/// Synchronized sum reduction with a barrier flag.
///
/// Returns `(sum, ready)` where `ready` is `true` once all active lanes
/// have contributed. On CPU this is always immediate; on GPU the barrier
/// maps to `__syncwarp`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn warp_sync_reduce_sum(data: &[f32], config: &ShuffleConfig) -> Result<(f32, bool)> {
    if data.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    let sum: f32 =
        (0..config.warp_size).filter(|&i| config.is_active(i)).map(|i| data[i as usize]).sum();
    Ok((sum, true))
}

// =========================================================================
// Vote / divergence helpers
// =========================================================================

/// Divergence ballot — returns a bitmask of lanes whose value differs
/// from lane 0's value.
///
/// Bit `i` is set when `data[i] != data[0]` and lane `i` is active.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn divergence_ballot(data: &[f32], config: &ShuffleConfig) -> Result<u32> {
    if data.len() < config.warp_size as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} < warp_size {}", data.len(), config.warp_size),
        }
        .into());
    }
    let ref_val = data[0];
    let mut mask = 0u32;
    for i in 0..config.warp_size {
        if config.is_active(i) && data[i as usize] != ref_val {
            mask |= 1 << i;
        }
    }
    Ok(mask)
}

/// Check whether all active lanes hold the same predicate value
/// (uniform branch).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `predicates` is too short.
pub fn uniform_branch_check(predicates: &[bool], config: &ShuffleConfig) -> Result<bool> {
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
    let mut first = None;
    for i in 0..config.warp_size {
        if config.is_active(i) {
            match first {
                None => first = Some(predicates[i as usize]),
                Some(v) if v != predicates[i as usize] => return Ok(false),
                _ => {}
            }
        }
    }
    Ok(true)
}

/// Count active lanes (equivalent to `__popc(__ballot_sync(mask, 1))`).
#[must_use]
pub fn active_lane_count(config: &ShuffleConfig) -> u32 {
    config.active_count()
}

/// Index of first active lane (equivalent to `__ffs(mask) - 1`).
///
/// Returns `None` if no lanes are active.
#[must_use]
pub fn first_active_lane(config: &ShuffleConfig) -> Option<u32> {
    (0..config.warp_size).find(|&i| config.is_active(i))
}

/// Population count of a ballot mask.
#[must_use]
pub fn popc(mask: u32) -> u32 {
    mask.count_ones()
}

// =========================================================================
// Matrix fragment operations (tensor core simulation)
// =========================================================================

/// Matrix layout for fragment load / store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Row-major storage.
    RowMajor,
    /// Column-major storage.
    ColMajor,
}

/// A small matrix fragment distributed across warp lanes.
///
/// Simulates CUDA WMMA fragments. Each element is conceptually owned by
/// one lane. For an M×N fragment, element `(r, c)` lives in lane
/// `r * N + c` (row-major assignment).
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFragment {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Flat data in row-major lane order.
    pub data: Vec<f32>,
}

impl MatrixFragment {
    /// Create a zero-filled fragment.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// Whether the fragment is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Load a matrix fragment from a buffer.
///
/// `buffer` is a contiguous row-major (or col-major) array of at least
/// `rows * cols` elements. `ld` is the leading dimension (stride between
/// rows for `RowMajor`, or between columns for `ColMajor`).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `buffer` is too small.
pub fn fragment_load(
    buffer: &[f32],
    rows: usize,
    cols: usize,
    ld: usize,
    layout: MatrixLayout,
) -> Result<MatrixFragment> {
    let required = match layout {
        MatrixLayout::RowMajor => {
            if rows == 0 {
                0
            } else {
                (rows - 1) * ld + cols
            }
        }
        MatrixLayout::ColMajor => {
            if cols == 0 {
                0
            } else {
                (cols - 1) * ld + rows
            }
        }
    };
    if buffer.len() < required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "fragment_load: buffer length {} < required {}",
                buffer.len(),
                required
            ),
        }
        .into());
    }
    let mut frag = MatrixFragment::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let src_idx = match layout {
                MatrixLayout::RowMajor => r * ld + c,
                MatrixLayout::ColMajor => c * ld + r,
            };
            frag.data[r * cols + c] = buffer[src_idx];
        }
    }
    Ok(frag)
}

/// Store a matrix fragment to a buffer.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `buffer` is too small.
pub fn fragment_store(
    frag: &MatrixFragment,
    buffer: &mut [f32],
    ld: usize,
    layout: MatrixLayout,
) -> Result<()> {
    let required = match layout {
        MatrixLayout::RowMajor => {
            if frag.rows == 0 {
                0
            } else {
                (frag.rows - 1) * ld + frag.cols
            }
        }
        MatrixLayout::ColMajor => {
            if frag.cols == 0 {
                0
            } else {
                (frag.cols - 1) * ld + frag.rows
            }
        }
    };
    if buffer.len() < required {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "fragment_store: buffer length {} < required {}",
                buffer.len(),
                required
            ),
        }
        .into());
    }
    for r in 0..frag.rows {
        for c in 0..frag.cols {
            let dst_idx = match layout {
                MatrixLayout::RowMajor => r * ld + c,
                MatrixLayout::ColMajor => c * ld + r,
            };
            buffer[dst_idx] = frag.data[r * frag.cols + c];
        }
    }
    Ok(())
}

/// Fill every element of a fragment with a scalar value.
pub fn fragment_fill(frag: &mut MatrixFragment, value: f32) {
    frag.data.fill(value);
}

/// Matrix multiply-accumulate: D = A · B + C.
///
/// A is `[M, K]`, B is `[K, N]`, C and D are `[M, N]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if dimensions are
/// incompatible.
pub fn fragment_mma(
    a: &MatrixFragment,
    b: &MatrixFragment,
    c: &MatrixFragment,
) -> Result<MatrixFragment> {
    if a.cols != b.rows {
        return Err(KernelError::InvalidArguments {
            reason: format!("fragment_mma: A.cols {} != B.rows {}", a.cols, b.rows),
        }
        .into());
    }
    if a.rows != c.rows || b.cols != c.cols {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "fragment_mma: C dims [{},{}] != expected [{},{}]",
                c.rows, c.cols, a.rows, b.cols
            ),
        }
        .into());
    }
    let m = a.rows;
    let n = b.cols;
    let k = a.cols;
    let mut d = MatrixFragment::zeros(m, n);
    for r in 0..m {
        for col in 0..n {
            let mut acc = c.data[r * n + col];
            for ki in 0..k {
                acc += a.data[r * k + ki] * b.data[ki * n + col];
            }
            d.data[r * n + col] = acc;
        }
    }
    Ok(d)
}

// =========================================================================
// Shuffle-based transpose
// =========================================================================

/// Transpose a 4×8 sub-matrix within a warp (32 lanes).
///
/// Input layout:  lane `r*8 + c` holds element at row `r`, col `c`
/// (4 rows × 8 cols = 32 elements).
/// Output layout: lane `c*4 + r` holds the same element, i.e. the result
/// is an 8×4 matrix.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length < 32.
pub fn shuffle_transpose_4x8(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let snapshot: Vec<f32> = data[..32].to_vec();
    for lane in 0u32..32 {
        if config.is_active(lane) {
            let row = lane / 8;
            let col = lane % 8;
            let src = col * 4 + row;
            data[lane as usize] = snapshot[src as usize];
        }
    }
    Ok(())
}

/// Transpose an 8×4 sub-matrix within a warp (32 lanes).
///
/// Input layout:  lane `r*4 + c` holds element at row `r`, col `c`
/// (8 rows × 4 cols = 32 elements).
/// Output layout: lane `c*8 + r` holds the same element, i.e. the result
/// is a 4×8 matrix.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` length < 32.
pub fn shuffle_transpose_8x4(data: &mut [f32], config: &ShuffleConfig) -> Result<()> {
    validate_data(data, config)?;
    let snapshot: Vec<f32> = data[..32].to_vec();
    for lane in 0u32..32 {
        if config.is_active(lane) {
            let row = lane / 4;
            let col = lane % 4;
            let src = col * 8 + row;
            data[lane as usize] = snapshot[src as usize];
        }
    }
    Ok(())
}

// =========================================================================
// Warp specialization
// =========================================================================

/// Role tag for warp specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpRole {
    /// Producer role (typically lower lane indices).
    Producer,
    /// Consumer role (typically higher lane indices).
    Consumer,
}

/// Configuration for splitting a warp into producer and consumer halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarpSpecConfig {
    /// Bitmask of producer lanes.
    pub producer_mask: u32,
    /// Bitmask of consumer lanes.
    pub consumer_mask: u32,
}

impl WarpSpecConfig {
    /// Split the warp at `split_lane`: lanes `[0, split_lane)` are
    /// producers and `[split_lane, 32)` are consumers.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `split_lane` is 0
    /// or ≥ 32.
    pub fn split_at(split_lane: u32) -> Result<Self> {
        if split_lane == 0 || split_lane >= WARP_SIZE {
            return Err(KernelError::InvalidArguments {
                reason: format!("split_lane must be in 1..31, got {split_lane}"),
            }
            .into());
        }
        let producer_mask = (1u32 << split_lane) - 1;
        let consumer_mask = !producer_mask;
        Ok(Self { producer_mask, consumer_mask })
    }

    /// Role of a given lane.
    #[must_use]
    pub fn role(&self, lane: u32) -> Option<WarpRole> {
        if lane >= WARP_SIZE {
            None
        } else if (self.producer_mask >> lane) & 1 == 1 {
            Some(WarpRole::Producer)
        } else if (self.consumer_mask >> lane) & 1 == 1 {
            Some(WarpRole::Consumer)
        } else {
            None
        }
    }

    /// Number of producer lanes.
    #[must_use]
    pub fn producer_count(&self) -> u32 {
        self.producer_mask.count_ones()
    }

    /// Number of consumer lanes.
    #[must_use]
    pub fn consumer_count(&self) -> u32 {
        self.consumer_mask.count_ones()
    }
}

/// Apply different transformations depending on lane role.
///
/// Producer lanes have `producer_fn` applied, consumer lanes have
/// `consumer_fn` applied. Other lanes are untouched.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn warp_specialize_map<F, G>(
    data: &mut [f32],
    spec: &WarpSpecConfig,
    producer_fn: F,
    consumer_fn: G,
) -> Result<()>
where
    F: Fn(f32) -> f32,
    G: Fn(f32) -> f32,
{
    if data.len() < WARP_SIZE as usize {
        return Err(KernelError::InvalidArguments {
            reason: format!("data length {} < WARP_SIZE {}", data.len(), WARP_SIZE),
        }
        .into());
    }
    for i in 0..WARP_SIZE {
        match spec.role(i) {
            Some(WarpRole::Producer) => data[i as usize] = producer_fn(data[i as usize]),
            Some(WarpRole::Consumer) => data[i as usize] = consumer_fn(data[i as usize]),
            None => {}
        }
    }
    Ok(())
}

/// Multi-stage pipeline within a warp.
///
/// `stages` is a list of transforms applied sequentially. Between each
/// stage a warp-wide shuffle-down by 1 is conceptually inserted so that
/// each stage's output feeds the next stage's input in a pipelined
/// fashion. On CPU this executes sequentially.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `data` is too short.
pub fn warp_pipeline_stages(
    data: &mut [f32],
    stages: &[fn(f32) -> f32],
    config: &ShuffleConfig,
) -> Result<()> {
    validate_data(data, config)?;
    for stage_fn in stages {
        for i in 0..config.warp_size {
            if config.is_active(i) {
                data[i as usize] = stage_fn(data[i as usize]);
            }
        }
    }
    Ok(())
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ShuffleConfig ──────────────────────────────────────────────

    #[test]
    fn test_shuffle_config_default() {
        let cfg = ShuffleConfig::new();
        assert_eq!(cfg.warp_size, 32);
        assert_eq!(cfg.active_mask, 0xFFFF_FFFF);
        assert_eq!(cfg.active_count(), 32);
    }

    #[test]
    fn test_shuffle_config_with_mask() {
        let cfg = ShuffleConfig::with_mask(0x0F).unwrap();
        assert_eq!(cfg.active_count(), 4);
        assert!(cfg.is_active(0));
        assert!(!cfg.is_active(4));
    }

    #[test]
    fn test_shuffle_config_zero_mask_rejected() {
        assert!(ShuffleConfig::with_mask(0).is_err());
    }

    #[test]
    fn test_shuffle_config_is_active_out_of_range() {
        let cfg = ShuffleConfig::new();
        assert!(!cfg.is_active(32));
        assert!(!cfg.is_active(100));
    }

    // ─── shuffle_xor ────────────────────────────────────────────────

    #[test]
    fn test_shuffle_xor_swap_adjacent() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_xor(&mut data, 1, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
        assert!((data[1] - 0.0).abs() < 1e-7);
        assert!((data[2] - 3.0).abs() < 1e-7);
        assert!((data[3] - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_shuffle_xor_identity() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let orig = data.clone();
        shuffle_xor(&mut data, 0, &cfg).unwrap();
        assert_eq!(data, orig);
    }

    #[test]
    fn test_shuffle_xor_reverse_half() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_xor(&mut data, 16, &cfg).unwrap();
        assert!((data[0] - 16.0).abs() < 1e-7);
        assert!((data[16] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_shuffle_xor_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b1111).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 10.0;
        data[1] = 20.0;
        data[2] = 30.0;
        data[3] = 40.0;
        data[8] = 99.0; // inactive
        shuffle_xor(&mut data, 1, &cfg).unwrap();
        assert!((data[0] - 20.0).abs() < 1e-7);
        assert!((data[1] - 10.0).abs() < 1e-7);
        assert!((data[8] - 99.0).abs() < 1e-7); // unchanged
    }

    #[test]
    fn test_shuffle_xor_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 16];
        assert!(shuffle_xor(&mut data, 1, &cfg).is_err());
    }

    // ─── shuffle_down ───────────────────────────────────────────────

    #[test]
    fn test_shuffle_down_by_one() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_down(&mut data, 1, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-7);
        assert!((data[30] - 31.0).abs() < 1e-7);
        // Lane 31 has no source above → keeps original
        assert!((data[31] - 31.0).abs() < 1e-7);
    }

    #[test]
    fn test_shuffle_down_by_16() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_down(&mut data, 16, &cfg).unwrap();
        assert!((data[0] - 16.0).abs() < 1e-7);
        assert!((data[15] - 31.0).abs() < 1e-7);
        // Lanes 16..31 keep originals
        assert!((data[16] - 16.0).abs() < 1e-7);
    }

    #[test]
    fn test_shuffle_down_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 8];
        assert!(shuffle_down(&mut data, 1, &cfg).is_err());
    }

    // ─── shuffle_up ─────────────────────────────────────────────────

    #[test]
    fn test_shuffle_up_by_one() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_up(&mut data, 1, &cfg).unwrap();
        // Lane 0 keeps original (no source below)
        assert!((data[0] - 0.0).abs() < 1e-7);
        assert!((data[1] - 0.0).abs() < 1e-7);
        assert!((data[31] - 30.0).abs() < 1e-7);
    }

    #[test]
    fn test_shuffle_up_by_16() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_up(&mut data, 16, &cfg).unwrap();
        // Lanes 0..15 keep originals
        assert!((data[0] - 0.0).abs() < 1e-7);
        assert!((data[15] - 15.0).abs() < 1e-7);
        // Lane 16 gets lane 0's value
        assert!((data[16] - 0.0).abs() < 1e-7);
        assert!((data[31] - 15.0).abs() < 1e-7);
    }

    #[test]
    fn test_shuffle_up_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 4];
        assert!(shuffle_up(&mut data, 1, &cfg).is_err());
    }

    // ─── shuffle_idx ────────────────────────────────────────────────

    #[test]
    fn test_shuffle_idx_broadcast() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_idx(&mut data, 7, &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 7.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_shuffle_idx_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b11).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 10.0;
        data[1] = 20.0;
        data[5] = 99.0;
        shuffle_idx(&mut data, 1, &cfg).unwrap();
        assert!((data[0] - 20.0).abs() < 1e-7);
        assert!((data[1] - 20.0).abs() < 1e-7);
        assert!((data[5] - 99.0).abs() < 1e-7); // unchanged
    }

    #[test]
    fn test_shuffle_idx_out_of_range() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        assert!(shuffle_idx(&mut data, 32, &cfg).is_err());
    }

    // ─── butterfly_reduce_sum ───────────────────────────────────────

    #[test]
    fn test_butterfly_reduce_sum_all() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        butterfly_reduce_sum(&mut data, &cfg).unwrap();
        let expected = (1..=32).sum::<u32>() as f32;
        for &v in &data[..32] {
            assert!((v - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn test_butterfly_reduce_sum_zeros() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        butterfly_reduce_sum(&mut data, &cfg).unwrap();
        for &v in &data[..32] {
            assert!(v.abs() < 1e-7);
        }
    }

    #[test]
    fn test_butterfly_reduce_sum_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0x0F).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[1] = 2.0;
        data[2] = 3.0;
        data[3] = 4.0;
        data[4] = 100.0; // inactive
        butterfly_reduce_sum(&mut data, &cfg).unwrap();
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[3] - 10.0).abs() < 1e-5);
        assert!((data[4] - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_butterfly_reduce_sum_negative() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| -(i as f32)).collect();
        butterfly_reduce_sum(&mut data, &cfg).unwrap();
        let expected: f32 = (0..32).map(|i| -(i as f32)).sum();
        assert!((data[0] - expected).abs() < 1e-2);
    }

    #[test]
    fn test_butterfly_reduce_sum_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 8];
        assert!(butterfly_reduce_sum(&mut data, &cfg).is_err());
    }

    // ─── butterfly_reduce_max ───────────────────────────────────────

    #[test]
    fn test_butterfly_reduce_max_all() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        butterfly_reduce_max(&mut data, &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 31.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_butterfly_reduce_max_negative() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| -100.0 + i as f32).collect();
        butterfly_reduce_max(&mut data, &cfg).unwrap();
        assert!((data[0] - (-69.0)).abs() < 1e-5);
    }

    #[test]
    fn test_butterfly_reduce_max_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b1010).unwrap();
        let mut data = vec![0.0f32; 32];
        data[1] = 3.0;
        data[3] = 7.0;
        data[0] = 999.0; // inactive
        butterfly_reduce_max(&mut data, &cfg).unwrap();
        assert!((data[1] - 7.0).abs() < 1e-5);
        assert!((data[3] - 7.0).abs() < 1e-5);
        assert!((data[0] - 999.0).abs() < 1e-5);
    }

    #[test]
    fn test_butterfly_reduce_max_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 4];
        assert!(butterfly_reduce_max(&mut data, &cfg).is_err());
    }

    // ─── butterfly_reduce_min ───────────────────────────────────────

    #[test]
    fn test_butterfly_reduce_min_all() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| (i + 10) as f32).collect();
        butterfly_reduce_min(&mut data, &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_butterfly_reduce_min_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b1100).unwrap();
        let mut data = vec![0.0f32; 32];
        data[2] = 5.0;
        data[3] = 2.0;
        data[0] = -999.0; // inactive
        butterfly_reduce_min(&mut data, &cfg).unwrap();
        assert!((data[2] - 2.0).abs() < 1e-5);
        assert!((data[3] - 2.0).abs() < 1e-5);
        assert!((data[0] - (-999.0)).abs() < 1e-5);
    }

    #[test]
    fn test_butterfly_reduce_min_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 4];
        assert!(butterfly_reduce_min(&mut data, &cfg).is_err());
    }

    // ─── halving_reduce_sum ─────────────────────────────────────────

    #[test]
    fn test_halving_reduce_sum_lane0() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        halving_reduce_sum(&mut data, &cfg).unwrap();
        let expected = (1..=32).sum::<u32>() as f32;
        assert!((data[0] - expected).abs() < 1e-3);
    }

    #[test]
    fn test_halving_reduce_sum_zeros() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        halving_reduce_sum(&mut data, &cfg).unwrap();
        assert!(data[0].abs() < 1e-7);
    }

    #[test]
    fn test_halving_reduce_sum_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 10];
        assert!(halving_reduce_sum(&mut data, &cfg).is_err());
    }

    // ─── halving_reduce_max ─────────────────────────────────────────

    #[test]
    fn test_halving_reduce_max_lane0() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        halving_reduce_max(&mut data, &cfg).unwrap();
        assert!((data[0] - 31.0).abs() < 1e-5);
    }

    #[test]
    fn test_halving_reduce_max_negative() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| -100.0 + i as f32).collect();
        halving_reduce_max(&mut data, &cfg).unwrap();
        assert!((data[0] - (-69.0)).abs() < 1e-5);
    }

    #[test]
    fn test_halving_reduce_max_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 2];
        assert!(halving_reduce_max(&mut data, &cfg).is_err());
    }

    // ─── shuffle_inclusive_scan ──────────────────────────────────────

    #[test]
    fn test_inclusive_scan_ones() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 32];
        shuffle_inclusive_scan(&mut data, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - (i + 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_inclusive_scan_varying() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        shuffle_inclusive_scan(&mut data, &cfg).unwrap();
        for i in 0..32usize {
            let expected = ((i + 1) * (i + 2)) as f32 / 2.0;
            assert!((data[i] - expected).abs() < 1e-2);
        }
    }

    #[test]
    fn test_inclusive_scan_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b10101).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[1] = 999.0; // inactive
        data[2] = 2.0;
        data[4] = 3.0;
        shuffle_inclusive_scan(&mut data, &cfg).unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 999.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[4] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_inclusive_scan_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 2];
        assert!(shuffle_inclusive_scan(&mut data, &cfg).is_err());
    }

    // ─── shuffle_exclusive_scan ─────────────────────────────────────

    #[test]
    fn test_exclusive_scan_ones() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 32];
        shuffle_exclusive_scan(&mut data, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_exclusive_scan_first_is_zero() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        shuffle_exclusive_scan(&mut data, &cfg).unwrap();
        assert!(data[0].abs() < 1e-7);
    }

    #[test]
    fn test_exclusive_scan_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 3];
        assert!(shuffle_exclusive_scan(&mut data, &cfg).is_err());
    }

    // ─── segmented_inclusive_scan ────────────────────────────────────

    #[test]
    fn test_segmented_scan_single_segment() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 32];
        let heads: Vec<bool> = (0..32).map(|i| i == 0).collect();
        segmented_inclusive_scan(&mut data, &heads, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - (i + 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_segmented_scan_two_segments() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 32];
        let heads: Vec<bool> = (0..32).map(|i| i == 0 || i == 16).collect();
        segmented_inclusive_scan(&mut data, &heads, &cfg).unwrap();
        // First segment: 1, 2, 3, ..., 16
        for i in 0..16 {
            assert!((data[i] - (i + 1) as f32).abs() < 1e-5);
        }
        // Second segment resets: 1, 2, 3, ..., 16
        for i in 16..32 {
            assert!((data[i] - (i - 16 + 1) as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_segmented_scan_every_lane_is_head() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![5.0f32; 32];
        let heads = vec![true; 32];
        segmented_inclusive_scan(&mut data, &heads, &cfg).unwrap();
        // Each lane is its own segment → identity
        for &v in &data[..32] {
            assert!((v - 5.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_segmented_scan_heads_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 32];
        let heads = vec![true; 10];
        assert!(segmented_inclusive_scan(&mut data, &heads, &cfg).is_err());
    }

    // ─── butterfly_exchange ─────────────────────────────────────────

    #[test]
    fn test_butterfly_exchange_stage0() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        butterfly_exchange(&mut data, 0, &cfg).unwrap();
        // Stage 0 → XOR mask 1 → swap adjacent pairs
        assert!((data[0] - 1.0).abs() < 1e-7);
        assert!((data[1] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_butterfly_exchange_stage1() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        butterfly_exchange(&mut data, 1, &cfg).unwrap();
        // Stage 1 → XOR mask 2 → swap within quads
        assert!((data[0] - 2.0).abs() < 1e-7);
        assert!((data[2] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_butterfly_exchange_stage4() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        butterfly_exchange(&mut data, 4, &cfg).unwrap();
        assert!((data[0] - 16.0).abs() < 1e-7);
        assert!((data[16] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_butterfly_exchange_out_of_range() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        assert!(butterfly_exchange(&mut data, 5, &cfg).is_err());
    }

    // ─── cross_warp_reduce_sum ──────────────────────────────────────

    #[test]
    fn test_cross_warp_reduce_sum_two_warps() {
        let cfg = ShuffleConfig::new();
        let w0 = vec![1.0f32; 32];
        let w1 = vec![2.0f32; 32];
        let mut out = vec![0.0f32; 32];
        cross_warp_reduce_sum(&[&w0, &w1], &mut out, &cfg).unwrap();
        for &v in &out[..32] {
            assert!((v - 3.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_cross_warp_reduce_sum_single_warp() {
        let cfg = ShuffleConfig::new();
        let w0: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 32];
        cross_warp_reduce_sum(&[&w0[..]], &mut out, &cfg).unwrap();
        assert_eq!(&out[..32], &w0[..32]);
    }

    #[test]
    fn test_cross_warp_reduce_sum_output_too_short() {
        let cfg = ShuffleConfig::new();
        let w0 = vec![1.0f32; 32];
        let mut out = vec![0.0f32; 8];
        assert!(cross_warp_reduce_sum(&[&w0[..]], &mut out, &cfg).is_err());
    }

    // ─── cross_warp_reduce_max ──────────────────────────────────────

    #[test]
    fn test_cross_warp_reduce_max_two_warps() {
        let cfg = ShuffleConfig::new();
        let w0: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let w1: Vec<f32> = (0..32).map(|i| (31 - i) as f32).collect();
        let mut out = vec![0.0f32; 32];
        cross_warp_reduce_max(&[&w0, &w1], &mut out, &cfg).unwrap();
        for &v in &out[..32] {
            assert!(v >= 15.0); // min of maxes
        }
    }

    #[test]
    fn test_cross_warp_reduce_max_empty_warps() {
        let cfg = ShuffleConfig::new();
        let mut out = vec![0.0f32; 32];
        assert!(cross_warp_reduce_max(&[], &mut out, &cfg).is_err());
    }

    #[test]
    fn test_cross_warp_reduce_max_output_too_short() {
        let cfg = ShuffleConfig::new();
        let w0 = vec![1.0f32; 32];
        let mut out = vec![0.0f32; 4];
        assert!(cross_warp_reduce_max(&[&w0[..]], &mut out, &cfg).is_err());
    }

    // ─── warp_allgather ─────────────────────────────────────────────

    #[test]
    fn test_allgather_full() {
        let cfg = ShuffleConfig::new();
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let gathered = warp_allgather(&data, &cfg).unwrap();
        assert_eq!(gathered, data[..32]);
    }

    #[test]
    fn test_allgather_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b11).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 10.0;
        data[1] = 20.0;
        data[5] = 50.0;
        let gathered = warp_allgather(&data, &cfg).unwrap();
        assert!((gathered[0] - 10.0).abs() < 1e-7);
        assert!((gathered[1] - 20.0).abs() < 1e-7);
        assert!((gathered[5] - 0.0).abs() < 1e-7); // inactive → 0
    }

    #[test]
    fn test_allgather_data_too_short() {
        let cfg = ShuffleConfig::new();
        let data = vec![0.0f32; 8];
        assert!(warp_allgather(&data, &cfg).is_err());
    }

    // ─── warp_scatter ───────────────────────────────────────────────

    #[test]
    fn test_scatter_to_all() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        warp_scatter(&mut data, 5, 0xFFFF_FFFF, &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 5.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_scatter_to_subset() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        data[0] = 42.0;
        warp_scatter(&mut data, 0, 0b1010, &cfg).unwrap();
        assert!((data[0] - 42.0).abs() < 1e-7); // not targeted
        assert!((data[1] - 42.0).abs() < 1e-7); // target
        assert!((data[2] - 0.0).abs() < 1e-7); // not targeted
        assert!((data[3] - 42.0).abs() < 1e-7); // target
    }

    #[test]
    fn test_scatter_src_out_of_range() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        assert!(warp_scatter(&mut data, 32, 0xFF, &cfg).is_err());
    }

    // ─── warp_sync_reduce_sum ───────────────────────────────────────

    #[test]
    fn test_sync_reduce_sum() {
        let cfg = ShuffleConfig::new();
        let data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let (sum, ready) = warp_sync_reduce_sum(&data, &cfg).unwrap();
        assert!(ready);
        assert!((sum - 528.0).abs() < 1e-3);
    }

    #[test]
    fn test_sync_reduce_sum_partial() {
        let cfg = ShuffleConfig::with_mask(0b111).unwrap();
        let mut data = vec![0.0f32; 32];
        data[0] = 10.0;
        data[1] = 20.0;
        data[2] = 30.0;
        let (sum, ready) = warp_sync_reduce_sum(&data, &cfg).unwrap();
        assert!(ready);
        assert!((sum - 60.0).abs() < 1e-5);
    }

    #[test]
    fn test_sync_reduce_sum_data_too_short() {
        let cfg = ShuffleConfig::new();
        let data = vec![0.0f32; 4];
        assert!(warp_sync_reduce_sum(&data, &cfg).is_err());
    }

    // ─── divergence_ballot ──────────────────────────────────────────

    #[test]
    fn test_divergence_ballot_all_same() {
        let cfg = ShuffleConfig::new();
        let data = vec![7.0f32; 32];
        let mask = divergence_ballot(&data, &cfg).unwrap();
        assert_eq!(mask, 0);
    }

    #[test]
    fn test_divergence_ballot_all_different() {
        let cfg = ShuffleConfig::new();
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mask = divergence_ballot(&data, &cfg).unwrap();
        // Lane 0 matches itself, all others diverge
        assert_eq!(mask, 0xFFFF_FFFE);
    }

    #[test]
    fn test_divergence_ballot_data_too_short() {
        let cfg = ShuffleConfig::new();
        let data = vec![0.0f32; 2];
        assert!(divergence_ballot(&data, &cfg).is_err());
    }

    // ─── uniform_branch_check ───────────────────────────────────────

    #[test]
    fn test_uniform_all_true() {
        let cfg = ShuffleConfig::new();
        let preds = vec![true; 32];
        assert!(uniform_branch_check(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_uniform_all_false() {
        let cfg = ShuffleConfig::new();
        let preds = vec![false; 32];
        assert!(uniform_branch_check(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_uniform_divergent() {
        let cfg = ShuffleConfig::new();
        let mut preds = vec![true; 32];
        preds[15] = false;
        assert!(!uniform_branch_check(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_uniform_partial_mask() {
        let cfg = ShuffleConfig::with_mask(0b11).unwrap();
        let mut preds = vec![false; 32]; // inactive lanes false
        preds[0] = true;
        preds[1] = true;
        assert!(uniform_branch_check(&preds, &cfg).unwrap());
    }

    #[test]
    fn test_uniform_preds_too_short() {
        let cfg = ShuffleConfig::new();
        let preds = vec![true; 10];
        assert!(uniform_branch_check(&preds, &cfg).is_err());
    }

    // ─── active_lane_count / first_active_lane / popc ───────────────

    #[test]
    fn test_active_lane_count_full() {
        let cfg = ShuffleConfig::new();
        assert_eq!(active_lane_count(&cfg), 32);
    }

    #[test]
    fn test_active_lane_count_partial() {
        let cfg = ShuffleConfig::with_mask(0b10101).unwrap();
        assert_eq!(active_lane_count(&cfg), 3);
    }

    #[test]
    fn test_first_active_lane_full() {
        let cfg = ShuffleConfig::new();
        assert_eq!(first_active_lane(&cfg), Some(0));
    }

    #[test]
    fn test_first_active_lane_offset() {
        let cfg = ShuffleConfig::with_mask(0b1000).unwrap();
        assert_eq!(first_active_lane(&cfg), Some(3));
    }

    #[test]
    fn test_popc() {
        assert_eq!(popc(0), 0);
        assert_eq!(popc(0xFFFF_FFFF), 32);
        assert_eq!(popc(0b1010_1010), 4);
        assert_eq!(popc(1), 1);
    }

    // ─── MatrixFragment / fragment ops ──────────────────────────────

    #[test]
    fn test_fragment_zeros() {
        let f = MatrixFragment::zeros(4, 4);
        assert_eq!(f.len(), 16);
        assert!(!f.is_empty());
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_fragment_fill() {
        let mut f = MatrixFragment::zeros(2, 3);
        fragment_fill(&mut f, 7.0);
        assert!(f.data.iter().all(|&v| (v - 7.0).abs() < 1e-7));
    }

    #[test]
    fn test_fragment_load_row_major() {
        let buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = fragment_load(&buf, 2, 3, 3, MatrixLayout::RowMajor).unwrap();
        assert_eq!(f.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_fragment_load_col_major() {
        // Col-major: columns are contiguous.
        // 2 rows, 3 cols, ld=2: buffer = [a00, a10, a01, a11, a02, a12]
        let buf = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let f = fragment_load(&buf, 2, 3, 2, MatrixLayout::ColMajor).unwrap();
        // Row-major output: [1,2,3, 4,5,6]
        assert_eq!(f.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_fragment_load_buffer_too_short() {
        let buf = vec![1.0, 2.0];
        assert!(fragment_load(&buf, 2, 3, 3, MatrixLayout::RowMajor).is_err());
    }

    #[test]
    fn test_fragment_store_row_major() {
        let f = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let mut buf = vec![0.0f32; 4];
        fragment_store(&f, &mut buf, 2, MatrixLayout::RowMajor).unwrap();
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fragment_store_col_major() {
        let f = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let mut buf = vec![0.0f32; 4];
        fragment_store(&f, &mut buf, 2, MatrixLayout::ColMajor).unwrap();
        // Col-major: [1, 3, 2, 4]
        assert_eq!(buf, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_fragment_store_buffer_too_short() {
        let f = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let mut buf = vec![0.0f32; 2];
        assert!(fragment_store(&f, &mut buf, 2, MatrixLayout::RowMajor).is_err());
    }

    #[test]
    fn test_fragment_mma_identity() {
        // A = I(2), B = [[1,2],[3,4]], C = 0 → D = B
        let a = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] };
        let b = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let c = MatrixFragment::zeros(2, 2);
        let d = fragment_mma(&a, &b, &c).unwrap();
        assert_eq!(d.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fragment_mma_accumulate() {
        let a = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] };
        let b = MatrixFragment { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] };
        let c = MatrixFragment { rows: 2, cols: 2, data: vec![10.0, 20.0, 30.0, 40.0] };
        let d = fragment_mma(&a, &b, &c).unwrap();
        assert_eq!(d.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_fragment_mma_2x3_3x2() {
        let a = MatrixFragment { rows: 2, cols: 3, data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] };
        let b = MatrixFragment { rows: 3, cols: 2, data: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0] };
        let c = MatrixFragment::zeros(2, 2);
        let d = fragment_mma(&a, &b, &c).unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((d.data[0] - 58.0).abs() < 1e-5);
        assert!((d.data[1] - 64.0).abs() < 1e-5);
        assert!((d.data[2] - 139.0).abs() < 1e-5);
        assert!((d.data[3] - 154.0).abs() < 1e-5);
    }

    #[test]
    fn test_fragment_mma_dimension_mismatch_k() {
        let a = MatrixFragment::zeros(2, 3);
        let b = MatrixFragment::zeros(4, 2); // K mismatch: 3 != 4
        let c = MatrixFragment::zeros(2, 2);
        assert!(fragment_mma(&a, &b, &c).is_err());
    }

    #[test]
    fn test_fragment_mma_dimension_mismatch_c() {
        let a = MatrixFragment::zeros(2, 3);
        let b = MatrixFragment::zeros(3, 2);
        let c = MatrixFragment::zeros(2, 3); // N mismatch: 3 != 2
        assert!(fragment_mma(&a, &b, &c).is_err());
    }

    // ─── shuffle_transpose_4x8 ─────────────────────────────────────

    #[test]
    fn test_transpose_4x8_basic() {
        let cfg = ShuffleConfig::new();
        // Input: 4 rows x 8 cols, row-major
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_transpose_4x8(&mut data, &cfg).unwrap();
        // After transpose: lane (r*8+c) gets value from (c*4+r)
        // Lane 0 (row=0, col=0) ← src (0*4+0)=0 → 0.0
        assert!((data[0] - 0.0).abs() < 1e-7);
        // Lane 1 (row=0, col=1) ← src (1*4+0)=4 → 4.0
        assert!((data[1] - 4.0).abs() < 1e-7);
        // Lane 8 (row=1, col=0) ← src (0*4+1)=1 → 1.0
        assert!((data[8] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_transpose_4x8_roundtrip() {
        let cfg = ShuffleConfig::new();
        let original: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut data = original.clone();
        // 4x8 → 8x4 → then transpose 8x4 → 4x8 should recover original
        shuffle_transpose_4x8(&mut data, &cfg).unwrap();
        shuffle_transpose_8x4(&mut data, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - original[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_transpose_4x8_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 16];
        assert!(shuffle_transpose_4x8(&mut data, &cfg).is_err());
    }

    // ─── shuffle_transpose_8x4 ─────────────────────────────────────

    #[test]
    fn test_transpose_8x4_basic() {
        let cfg = ShuffleConfig::new();
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        shuffle_transpose_8x4(&mut data, &cfg).unwrap();
        // Lane 0 (row=0, col=0) ← src (0*8+0)=0 → 0.0
        assert!((data[0] - 0.0).abs() < 1e-7);
        // Lane 1 (row=0, col=1) ← src (1*8+0)=8 → 8.0
        assert!((data[1] - 8.0).abs() < 1e-7);
        // Lane 4 (row=1, col=0) ← src (0*8+1)=1 → 1.0
        assert!((data[4] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_transpose_8x4_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 16];
        assert!(shuffle_transpose_8x4(&mut data, &cfg).is_err());
    }

    // ─── WarpSpecConfig ─────────────────────────────────────────────

    #[test]
    fn test_spec_config_split_at_16() {
        let spec = WarpSpecConfig::split_at(16).unwrap();
        assert_eq!(spec.producer_count(), 16);
        assert_eq!(spec.consumer_count(), 16);
        assert_eq!(spec.role(0), Some(WarpRole::Producer));
        assert_eq!(spec.role(15), Some(WarpRole::Producer));
        assert_eq!(spec.role(16), Some(WarpRole::Consumer));
        assert_eq!(spec.role(31), Some(WarpRole::Consumer));
    }

    #[test]
    fn test_spec_config_split_at_1() {
        let spec = WarpSpecConfig::split_at(1).unwrap();
        assert_eq!(spec.producer_count(), 1);
        assert_eq!(spec.consumer_count(), 31);
    }

    #[test]
    fn test_spec_config_split_at_0_rejected() {
        assert!(WarpSpecConfig::split_at(0).is_err());
    }

    #[test]
    fn test_spec_config_split_at_32_rejected() {
        assert!(WarpSpecConfig::split_at(32).is_err());
    }

    #[test]
    fn test_spec_config_role_out_of_range() {
        let spec = WarpSpecConfig::split_at(16).unwrap();
        assert_eq!(spec.role(32), None);
    }

    // ─── warp_specialize_map ────────────────────────────────────────

    #[test]
    fn test_specialize_map_double_negate() {
        let spec = WarpSpecConfig::split_at(16).unwrap();
        let mut data = vec![1.0f32; 32];
        warp_specialize_map(&mut data, &spec, |v| v * 2.0, |v| -v).unwrap();
        // Producers (0..16) doubled
        for &v in &data[..16] {
            assert!((v - 2.0).abs() < 1e-7);
        }
        // Consumers (16..32) negated
        for &v in &data[16..32] {
            assert!((v - (-1.0)).abs() < 1e-7);
        }
    }

    #[test]
    fn test_specialize_map_data_too_short() {
        let spec = WarpSpecConfig::split_at(16).unwrap();
        let mut data = vec![1.0f32; 8];
        assert!(warp_specialize_map(&mut data, &spec, |v| v, |v| v).is_err());
    }

    // ─── warp_pipeline_stages ───────────────────────────────────────

    #[test]
    fn test_pipeline_single_stage() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![2.0f32; 32];
        warp_pipeline_stages(&mut data, &[|v| v * 3.0], &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 6.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_pipeline_two_stages() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![1.0f32; 32];
        warp_pipeline_stages(&mut data, &[|v| v + 1.0, |v| v * 2.0], &cfg).unwrap();
        // stage 1: 1+1=2, stage 2: 2*2=4
        for &v in &data[..32] {
            assert!((v - 4.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_pipeline_zero_stages() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![5.0f32; 32];
        warp_pipeline_stages(&mut data, &[], &cfg).unwrap();
        for &v in &data[..32] {
            assert!((v - 5.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_pipeline_data_too_short() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 4];
        assert!(warp_pipeline_stages(&mut data, &[|v| v], &cfg).is_err());
    }

    // ─── Consistency / integration tests ────────────────────────────

    #[test]
    fn test_butterfly_and_halving_sum_agree_on_lane0() {
        let cfg = ShuffleConfig::new();
        let values: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let mut bf = values.clone();
        butterfly_reduce_sum(&mut bf, &cfg).unwrap();
        let mut hf = values;
        halving_reduce_sum(&mut hf, &cfg).unwrap();
        assert!((bf[0] - hf[0]).abs() < 1e-3);
    }

    #[test]
    fn test_butterfly_max_and_halving_max_agree_on_lane0() {
        let cfg = ShuffleConfig::new();
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.7).collect();
        let mut bf = values.clone();
        butterfly_reduce_max(&mut bf, &cfg).unwrap();
        let mut hf = values;
        halving_reduce_max(&mut hf, &cfg).unwrap();
        assert!((bf[0] - hf[0]).abs() < 1e-5);
    }

    #[test]
    fn test_inclusive_scan_last_equals_butterfly_sum() {
        let cfg = ShuffleConfig::new();
        let values: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let mut scan = values.clone();
        shuffle_inclusive_scan(&mut scan, &cfg).unwrap();
        let mut bf = values;
        butterfly_reduce_sum(&mut bf, &cfg).unwrap();
        assert!((scan[31] - bf[0]).abs() < 1e-2);
    }

    #[test]
    fn test_exclusive_scan_shift_of_inclusive() {
        let cfg = ShuffleConfig::new();
        let values = vec![3.0f32; 32];
        let mut incl = values.clone();
        shuffle_inclusive_scan(&mut incl, &cfg).unwrap();
        let mut excl = values;
        shuffle_exclusive_scan(&mut excl, &cfg).unwrap();
        for i in 0..32 {
            assert!((excl[i] - (incl[i] - 3.0)).abs() < 1e-4);
        }
    }

    #[test]
    fn test_shuffle_xor_is_self_inverse() {
        let cfg = ShuffleConfig::new();
        let original: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut data = original.clone();
        shuffle_xor(&mut data, 5, &cfg).unwrap();
        shuffle_xor(&mut data, 5, &cfg).unwrap();
        for i in 0..32 {
            assert!((data[i] - original[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_divergence_and_uniform_consistent() {
        let cfg = ShuffleConfig::new();
        let data = vec![5.0f32; 32];
        let div_mask = divergence_ballot(&data, &cfg).unwrap();
        let preds: Vec<bool> = data.iter().map(|&v| v > 0.0).collect();
        let uniform = uniform_branch_check(&preds, &cfg).unwrap();
        assert_eq!(div_mask, 0); // all same → no divergence
        assert!(uniform); // all same predicate
    }

    #[test]
    fn test_fragment_load_store_roundtrip() {
        let buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let f = fragment_load(&buf, 3, 3, 3, MatrixLayout::RowMajor).unwrap();
        let mut out = vec![0.0f32; 9];
        fragment_store(&f, &mut out, 3, MatrixLayout::RowMajor).unwrap();
        assert_eq!(out, buf);
    }

    #[test]
    fn test_cross_warp_sum_matches_manual() {
        let cfg = ShuffleConfig::new();
        let w0: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let w1: Vec<f32> = (0..32).map(|i| (i * 2) as f32).collect();
        let mut out = vec![0.0f32; 32];
        cross_warp_reduce_sum(&[&w0, &w1], &mut out, &cfg).unwrap();
        for i in 0..32 {
            let expected = i as f32 + (i * 2) as f32;
            assert!((out[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_scatter_then_reduce_pattern() {
        let cfg = ShuffleConfig::new();
        let mut data = vec![0.0f32; 32];
        data[0] = 10.0;
        // Scatter lane 0 to all, then reduce — should get 10*32
        warp_scatter(&mut data, 0, 0xFFFF_FFFF, &cfg).unwrap();
        let mut reduced = data.clone();
        butterfly_reduce_sum(&mut reduced, &cfg).unwrap();
        assert!((reduced[0] - 320.0).abs() < 1e-2);
    }
}
