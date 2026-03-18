//! CUDA attention mask generation and application kernels with CPU fallback.
//!
//! # Mask types
//!
//! - **Causal mask** — lower-triangular mask for autoregressive decoding.
//!   Position `(i, j)` is visible iff `j <= i`.
//! - **Padding mask** — masks padding tokens using per-sequence lengths.
//! - **Combined mask** — intersection of causal and padding masks.
//! - **Sliding window mask** — Mistral-style local attention where each
//!   token attends to at most `window_size` preceding tokens.
//! - **Block sparse mask** — BigBird-style block-sparse attention with
//!   fixed block size.
//! - **ALiBi mask** — Attention with Linear Biases (Press et al., 2022),
//!   adds position-dependent bias `−m · |i − j|` to scores.
//! - **Prefix mask** — prefix LM: prefix tokens attend to all positions,
//!   suffix tokens use causal masking.
//!
//! # Application modes
//!
//! - **In-place** — overwrites masked positions in the score tensor.
//! - **Additive** — adds `−∞` (actually `NEG_INF`) to masked positions.
//! - **Multiplicative** — multiplies scores by `0.0` (masked) or `1.0`.
//!
//! # Kernel strategy
//!
//! Mask generation kernels use grid-stride loops with 256 threads per block.
//! Each thread computes one `(row, col)` element of the mask matrix.
//! Application kernels operate element-wise on pre-allocated score buffers.
//!
//! # CPU fallback
//!
//! Every public function has a pure-Rust implementation used when the `gpu`
//! / `cuda` features are not active, and for correctness testing.

use bitnet_common::{KernelError, Result};

/// Sentinel value for masked positions in additive masking.
///
/// Using `−1e9` instead of `f32::NEG_INFINITY` to avoid NaN propagation in
/// downstream softmax when all positions in a row are masked.
pub const NEG_INF: f32 = -1e9;

// ── CUDA kernel source ───────────────────────────────────────────────

/// Inline CUDA C source for attention mask kernels.
///
/// Contains kernels:
/// - `causal_mask_f32` — generates lower-triangular causal mask
/// - `padding_mask_f32` — generates padding mask from sequence lengths
/// - `combined_mask_f32` — causal + padding intersection
/// - `sliding_window_mask_f32` — local attention window mask
/// - `block_sparse_mask_f32` — block-sparse attention mask
/// - `alibi_mask_f32` — ALiBi position bias mask
/// - `prefix_mask_f32` — prefix LM mask
/// - `apply_mask_additive_f32` — additive masking (add NEG_INF)
/// - `apply_mask_multiplicative_f32` — multiplicative masking (mul 0/1)
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const ATTENTION_MASK_KERNEL_SRC: &str = r#"
extern "C" __global__ void causal_mask_f32(
    float* __restrict__ mask,
    int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int row = i / seq_len;
        int col = i % seq_len;
        mask[i] = (col <= row) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void padding_mask_f32(
    float* __restrict__ mask,
    const int* __restrict__ seq_lengths,
    int batch_size,
    int max_seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * max_seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int b = i / max_seq_len;
        int pos = i % max_seq_len;
        mask[i] = (pos < seq_lengths[b]) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void combined_mask_f32(
    float* __restrict__ mask,
    const int* __restrict__ seq_lengths,
    int batch_size,
    int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int b = i / (seq_len * seq_len);
        int rem = i % (seq_len * seq_len);
        int row = rem / seq_len;
        int col = rem % seq_len;
        int valid_len = seq_lengths[b];
        int causal_ok = (col <= row) ? 1 : 0;
        int padding_ok = (col < valid_len && row < valid_len) ? 1 : 0;
        mask[i] = (float)(causal_ok & padding_ok);
    }
}

extern "C" __global__ void sliding_window_mask_f32(
    float* __restrict__ mask,
    int seq_len,
    int window_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int row = i / seq_len;
        int col = i % seq_len;
        int causal_ok = (col <= row) ? 1 : 0;
        int window_ok = ((row - col) < window_size) ? 1 : 0;
        mask[i] = (float)(causal_ok & window_ok);
    }
}

extern "C" __global__ void block_sparse_mask_f32(
    float* __restrict__ mask,
    int seq_len,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int row = i / seq_len;
        int col = i % seq_len;
        int row_block = row / block_size;
        int col_block = col / block_size;
        mask[i] = (row_block == col_block) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void alibi_mask_f32(
    float* __restrict__ mask,
    int seq_len,
    int n_heads,
    const float* __restrict__ slopes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * seq_len * seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int h = i / (seq_len * seq_len);
        int rem = i % (seq_len * seq_len);
        int row = rem / seq_len;
        int col = rem % seq_len;
        int dist = (row >= col) ? (row - col) : (col - row);
        mask[i] = -slopes[h] * (float)dist;
    }
}

extern "C" __global__ void prefix_mask_f32(
    float* __restrict__ mask,
    int seq_len,
    int prefix_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int row = i / seq_len;
        int col = i % seq_len;
        // Prefix rows attend to all positions; suffix rows are causal.
        int ok;
        if (row < prefix_len) {
            ok = 1;
        } else {
            ok = (col <= row) ? 1 : 0;
        }
        mask[i] = (float)ok;
    }
}

extern "C" __global__ void apply_mask_additive_f32(
    float* __restrict__ scores,
    const float* __restrict__ mask,
    int n,
    float neg_inf_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        if (mask[i] == 0.0f) {
            scores[i] = neg_inf_val;
        }
    }
}

extern "C" __global__ void apply_mask_multiplicative_f32(
    float* __restrict__ scores,
    const float* __restrict__ mask,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        scores[i] *= mask[i];
    }
}
"#;

// ── Configuration types ──────────────────────────────────────────────

/// Launch configuration for attention mask kernels.
#[derive(Debug, Clone)]
pub struct AttentionMaskConfig {
    /// Sequence length (Q dimension).
    pub seq_len: usize,
    /// Threads per CUDA block (default 256).
    pub threads_per_block: u32,
}

impl AttentionMaskConfig {
    /// Create a mask configuration for the given sequence length.
    pub fn new(seq_len: usize) -> Self {
        Self { seq_len, threads_per_block: 256 }
    }

    /// CUDA grid dimensions for a `seq_len × seq_len` mask.
    pub fn grid_dim(&self) -> (u32, u32, u32) {
        let total = (self.seq_len * self.seq_len) as u32;
        let blocks = total.div_ceil(self.threads_per_block);
        (blocks.min(65_535), 1, 1)
    }

    /// CUDA block dimensions.
    pub fn block_dim(&self) -> (u32, u32, u32) {
        (self.threads_per_block, 1, 1)
    }
}

/// Configuration for sliding window attention masks.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Sequence length.
    pub seq_len: usize,
    /// Window size — each token attends to this many preceding tokens
    /// (including itself).
    pub window_size: usize,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
}

impl SlidingWindowConfig {
    /// Create a sliding window mask configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `window_size` is zero.
    pub fn new(seq_len: usize, window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "sliding window size must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { seq_len, window_size, threads_per_block: 256 })
    }
}

/// Configuration for block-sparse attention masks.
#[derive(Debug, Clone)]
pub struct BlockSparseConfig {
    /// Sequence length.
    pub seq_len: usize,
    /// Block size for the sparse pattern.
    pub block_size: usize,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
}

impl BlockSparseConfig {
    /// Create a block-sparse mask configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `block_size` is zero.
    pub fn new(seq_len: usize, block_size: usize) -> Result<Self> {
        if block_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "block size must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { seq_len, block_size, threads_per_block: 256 })
    }
}

/// Configuration for ALiBi mask generation.
#[derive(Debug, Clone)]
pub struct AlibiConfig {
    /// Sequence length.
    pub seq_len: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
}

impl AlibiConfig {
    /// Create an ALiBi mask configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `n_heads` is zero.
    pub fn new(seq_len: usize, n_heads: usize) -> Result<Self> {
        if n_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "number of heads must be non-zero".into(),
            }
            .into());
        }
        Ok(Self { seq_len, n_heads, threads_per_block: 256 })
    }
}

/// Configuration for prefix LM masks.
#[derive(Debug, Clone)]
pub struct PrefixMaskConfig {
    /// Total sequence length (prefix + suffix).
    pub seq_len: usize,
    /// Length of the prefix region (bidirectional attention).
    pub prefix_len: usize,
    /// Threads per CUDA block.
    pub threads_per_block: u32,
}

impl PrefixMaskConfig {
    /// Create a prefix mask configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if `prefix_len > seq_len`.
    pub fn new(seq_len: usize, prefix_len: usize) -> Result<Self> {
        if prefix_len > seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!("prefix_len ({prefix_len}) must not exceed seq_len ({seq_len})"),
            }
            .into());
        }
        Ok(Self { seq_len, prefix_len, threads_per_block: 256 })
    }
}

// ── ALiBi slope computation ──────────────────────────────────────────

/// Compute ALiBi slopes for `n_heads` attention heads.
///
/// Uses the geometric sequence from Press et al. (2022):
/// `slope_i = 2^(−8·i / n_heads)` for `i` in `1..=n_heads`.
///
/// When `n_heads` is not a power of 2, the slopes are interpolated
/// following the original paper's prescription.
pub fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    if n_heads == 0 {
        return vec![];
    }
    // Find closest power of 2 <= n_heads
    let closest_power_of_2 = 1usize << (usize::BITS - 1 - n_heads.leading_zeros());

    let base = 2.0_f64.powf(-(8.0 / closest_power_of_2 as f64));
    let mut slopes = Vec::with_capacity(n_heads);

    if closest_power_of_2 == n_heads {
        // Power-of-2 case: geometric sequence
        for i in 1..=n_heads {
            slopes.push(base.powi(i as i32) as f32);
        }
    } else {
        // Non-power-of-2: interleave two geometric sequences
        let extra_base = 2.0_f64.powf(-(8.0 / (2 * closest_power_of_2) as f64));
        for i in 1..=closest_power_of_2 {
            slopes.push(base.powi(i as i32) as f32);
        }
        for i in 1..=(n_heads - closest_power_of_2) {
            slopes.push(extra_base.powi((2 * i - 1) as i32) as f32);
        }
    }
    slopes
}

// ── GPU launch stubs ─────────────────────────────────────────────────

/// Launch the CUDA causal mask kernel.
///
/// Writes a `seq_len × seq_len` lower-triangular mask into `output`.
///
/// # Errors
///
/// Returns [`KernelError::LaunchFailed`] if the kernel launch fails.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_causal_mask(_config: &AttentionMaskConfig, _output: &mut [f32]) -> Result<()> {
    Err(KernelError::LaunchFailed {
        kernel: "causal_mask_f32".into(),
        reason: "CUDA runtime not available — use causal_mask CPU fallback".into(),
    }
    .into())
}

/// Launch the CUDA sliding window mask kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_sliding_window_mask(
    _config: &SlidingWindowConfig,
    _output: &mut [f32],
) -> Result<()> {
    Err(KernelError::LaunchFailed {
        kernel: "sliding_window_mask_f32".into(),
        reason: "CUDA runtime not available — use sliding_window_mask CPU fallback".into(),
    }
    .into())
}

/// Launch the CUDA ALiBi mask kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_alibi_mask(
    _config: &AlibiConfig,
    _slopes: &[f32],
    _output: &mut [f32],
) -> Result<()> {
    Err(KernelError::LaunchFailed {
        kernel: "alibi_mask_f32".into(),
        reason: "CUDA runtime not available — use alibi_mask CPU fallback".into(),
    }
    .into())
}

/// Launch the CUDA additive mask application kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_apply_mask_additive(_scores: &mut [f32], _mask: &[f32]) -> Result<()> {
    Err(KernelError::LaunchFailed {
        kernel: "apply_mask_additive_f32".into(),
        reason: "CUDA runtime not available — use apply_mask_additive CPU fallback".into(),
    }
    .into())
}

/// Launch the CUDA multiplicative mask application kernel.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_apply_mask_multiplicative(_scores: &mut [f32], _mask: &[f32]) -> Result<()> {
    Err(KernelError::LaunchFailed {
        kernel: "apply_mask_multiplicative_f32".into(),
        reason: "CUDA runtime not available — use apply_mask_multiplicative CPU fallback".into(),
    }
    .into())
}

// ── CPU fallback implementations ─────────────────────────────────────

/// Generate a causal (lower-triangular) attention mask.
///
/// Output is a flat `seq_len × seq_len` buffer. `mask[i * seq_len + j]` is
/// `1.0` if `j <= i`, else `0.0`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `output` is too small.
pub fn causal_mask(seq_len: usize, output: &mut [f32]) -> Result<()> {
    let total = seq_len
        .checked_mul(seq_len)
        .ok_or_else(|| KernelError::InvalidArguments { reason: "seq_len overflow".into() })?;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for row in 0..seq_len {
        for col in 0..seq_len {
            output[row * seq_len + col] = if col <= row { 1.0 } else { 0.0 };
        }
    }
    Ok(())
}

/// Generate a padding mask from per-sequence lengths.
///
/// Output shape: `batch_size × max_seq_len` (flat).
/// `mask[b * max_seq_len + pos]` is `1.0` if `pos < seq_lengths[b]`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes mismatch.
pub fn padding_mask(seq_lengths: &[usize], max_seq_len: usize, output: &mut [f32]) -> Result<()> {
    let batch_size = seq_lengths.len();
    let total = batch_size * max_seq_len;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for (b, &len) in seq_lengths.iter().enumerate() {
        for pos in 0..max_seq_len {
            output[b * max_seq_len + pos] = if pos < len { 1.0 } else { 0.0 };
        }
    }
    Ok(())
}

/// Generate a combined causal + padding attention mask.
///
/// Output shape: `batch_size × seq_len × seq_len` (flat).
/// A position is unmasked iff `col <= row` AND both `row` and `col` are
/// within the valid length for the batch element.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if buffer sizes mismatch.
pub fn combined_mask(seq_lengths: &[usize], seq_len: usize, output: &mut [f32]) -> Result<()> {
    let batch_size = seq_lengths.len();
    let total = batch_size * seq_len * seq_len;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for (b, &valid_len) in seq_lengths.iter().enumerate() {
        let offset = b * seq_len * seq_len;
        for row in 0..seq_len {
            for col in 0..seq_len {
                let causal_ok = col <= row;
                let padding_ok = col < valid_len && row < valid_len;
                output[offset + row * seq_len + col] =
                    if causal_ok && padding_ok { 1.0 } else { 0.0 };
            }
        }
    }
    Ok(())
}

/// Generate a sliding window attention mask (Mistral-style).
///
/// Output is a flat `seq_len × seq_len` buffer. Position `(i, j)` is
/// unmasked iff `j <= i` (causal) AND `i − j < window_size`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on size mismatch or zero window.
pub fn sliding_window_mask(seq_len: usize, window_size: usize, output: &mut [f32]) -> Result<()> {
    if window_size == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "window_size must be non-zero".into(),
        }
        .into());
    }
    let total = seq_len * seq_len;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for row in 0..seq_len {
        for col in 0..seq_len {
            let visible = col <= row && (row - col) < window_size;
            output[row * seq_len + col] = if visible { 1.0 } else { 0.0 };
        }
    }
    Ok(())
}

/// Generate a block-sparse attention mask (BigBird-style).
///
/// Output is a flat `seq_len × seq_len` buffer. Position `(i, j)` is
/// unmasked iff `i / block_size == j / block_size`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on size mismatch or zero block.
pub fn block_sparse_mask(seq_len: usize, block_size: usize, output: &mut [f32]) -> Result<()> {
    if block_size == 0 {
        return Err(
            KernelError::InvalidArguments { reason: "block_size must be non-zero".into() }.into()
        );
    }
    let total = seq_len * seq_len;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for row in 0..seq_len {
        for col in 0..seq_len {
            let same_block = row / block_size == col / block_size;
            output[row * seq_len + col] = if same_block { 1.0 } else { 0.0 };
        }
    }
    Ok(())
}

/// Generate an ALiBi attention bias mask.
///
/// Output shape: `n_heads × seq_len × seq_len` (flat).
/// `mask[h * seq_len² + i * seq_len + j] = −slopes[h] · |i − j|`.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] on size mismatch.
pub fn alibi_mask(seq_len: usize, slopes: &[f32], output: &mut [f32]) -> Result<()> {
    let n_heads = slopes.len();
    let total = n_heads * seq_len * seq_len;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for (h, &slope) in slopes.iter().enumerate() {
        let h_offset = h * seq_len * seq_len;
        for row in 0..seq_len {
            for col in 0..seq_len {
                let dist = row.abs_diff(col);
                output[h_offset + row * seq_len + col] = -slope * dist as f32;
            }
        }
    }
    Ok(())
}

/// Apply mask to attention scores in-place.
///
/// Masked positions (`mask == 0.0`) are set to [`NEG_INF`]; unmasked
/// positions are left unchanged.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ.
pub fn apply_mask_to_scores(scores: &mut [f32], mask: &[f32]) -> Result<()> {
    if scores.len() != mask.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("scores length {} != mask length {}", scores.len(), mask.len()),
        }
        .into());
    }
    for (s, &m) in scores.iter_mut().zip(mask.iter()) {
        if m == 0.0 {
            *s = NEG_INF;
        }
    }
    Ok(())
}

/// Additive masking — adds [`NEG_INF`] to masked positions.
///
/// For positions where `mask == 0.0`, the score is replaced with
/// [`NEG_INF`]. Unmasked positions are unchanged.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ.
pub fn apply_mask_additive(scores: &mut [f32], mask: &[f32]) -> Result<()> {
    if scores.len() != mask.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("scores length {} != mask length {}", scores.len(), mask.len()),
        }
        .into());
    }
    for (s, &m) in scores.iter_mut().zip(mask.iter()) {
        if m == 0.0 {
            *s = NEG_INF;
        }
    }
    Ok(())
}

/// Multiplicative masking — multiply scores by 0/1 mask.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if lengths differ.
pub fn apply_mask_multiplicative(scores: &mut [f32], mask: &[f32]) -> Result<()> {
    if scores.len() != mask.len() {
        return Err(KernelError::InvalidArguments {
            reason: format!("scores length {} != mask length {}", scores.len(), mask.len()),
        }
        .into());
    }
    for (s, &m) in scores.iter_mut().zip(mask.iter()) {
        *s *= m;
    }
    Ok(())
}

/// Generate a prefix LM mask.
///
/// Output is a flat `seq_len × seq_len` buffer.
/// - Rows `0..prefix_len` (prefix) attend to **all** positions (`1.0`).
/// - Rows `prefix_len..seq_len` (suffix) use causal masking (`col <= row`).
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `prefix_len > seq_len` or
/// output buffer too small.
pub fn create_prefix_mask(seq_len: usize, prefix_len: usize, output: &mut [f32]) -> Result<()> {
    if prefix_len > seq_len {
        return Err(KernelError::InvalidArguments {
            reason: format!("prefix_len ({prefix_len}) must not exceed seq_len ({seq_len})"),
        }
        .into());
    }
    let total = seq_len * seq_len;
    if output.len() < total {
        return Err(KernelError::InvalidArguments {
            reason: format!("output length {} < required {total}", output.len()),
        }
        .into());
    }
    for row in 0..seq_len {
        for col in 0..seq_len {
            let visible = if row < prefix_len {
                true // prefix attends to everything
            } else {
                col <= row // suffix is causal
            };
            output[row * seq_len + col] = if visible { 1.0 } else { 0.0 };
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────

    fn mask_buf(size: usize) -> Vec<f32> {
        vec![0.0; size]
    }

    fn print_mask(mask: &[f32], rows: usize, cols: usize) {
        for r in 0..rows {
            let row: Vec<String> =
                (0..cols).map(|c| format!("{:5.1}", mask[r * cols + c])).collect();
            eprintln!("{}", row.join(" "));
        }
    }

    // ── Causal mask ─────────────────────────────────────────────────

    #[test]
    fn test_causal_mask_seq1() {
        let mut out = mask_buf(1);
        causal_mask(1, &mut out).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_causal_mask_seq2() {
        let mut out = mask_buf(4);
        causal_mask(2, &mut out).unwrap();
        // Row 0: [1, 0]
        // Row 1: [1, 1]
        assert_eq!(out, vec![1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_causal_mask_seq4() {
        let mut out = mask_buf(16);
        causal_mask(4, &mut out).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_causal_mask_seq8() {
        let mut out = mask_buf(64);
        causal_mask(8, &mut out).unwrap();
        for row in 0..8 {
            for col in 0..8 {
                let expected = if col <= row { 1.0 } else { 0.0 };
                assert_eq!(out[row * 8 + col], expected, "mismatch at ({row}, {col})");
            }
        }
    }

    #[test]
    fn test_causal_mask_seq64() {
        let n = 64;
        let mut out = mask_buf(n * n);
        causal_mask(n, &mut out).unwrap();
        for row in 0..n {
            for col in 0..n {
                let expected = if col <= row { 1.0 } else { 0.0 };
                assert_eq!(out[row * n + col], expected, "mismatch at ({row}, {col})");
            }
        }
    }

    #[test]
    fn test_causal_mask_seq512() {
        let n = 512;
        let mut out = mask_buf(n * n);
        causal_mask(n, &mut out).unwrap();
        // Spot-check diagonal and anti-diagonal
        for i in 0..n {
            assert_eq!(out[i * n + i], 1.0, "diagonal ({i},{i})");
        }
        for i in 0..(n - 1) {
            assert_eq!(out[i * n + (i + 1)], 0.0, "super-diagonal ({i},{})", i + 1);
        }
    }

    #[test]
    fn test_causal_mask_seq0() {
        let mut out = mask_buf(0);
        causal_mask(0, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_causal_mask_lower_triangular_sum() {
        let n = 16;
        let mut out = mask_buf(n * n);
        causal_mask(n, &mut out).unwrap();
        let sum: f32 = out.iter().sum();
        // Sum of 1..=n = n*(n+1)/2
        let expected = (n * (n + 1) / 2) as f32;
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_causal_mask_buffer_too_small() {
        let mut out = mask_buf(3);
        let result = causal_mask(2, &mut out);
        assert!(result.is_err());
    }

    // ── Padding mask ────────────────────────────────────────────────

    #[test]
    fn test_padding_mask_uniform() {
        let lengths = [4, 4, 4];
        let mut out = mask_buf(12);
        padding_mask(&lengths, 4, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_padding_mask_variable() {
        let lengths = [2, 4, 1];
        let mut out = mask_buf(12);
        padding_mask(&lengths, 4, &mut out).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 1.0, 0.0, 0.0, // batch 0: len=2
            1.0, 1.0, 1.0, 1.0, // batch 1: len=4
            1.0, 0.0, 0.0, 0.0, // batch 2: len=1
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_padding_mask_zero_length() {
        let lengths = [0, 3];
        let mut out = mask_buf(8);
        padding_mask(&lengths, 4, &mut out).unwrap();
        // First batch: all zeros
        assert!(out[0..4].iter().all(|&v| v == 0.0));
        // Second batch: first 3 are 1
        assert_eq!(&out[4..8], &[1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_padding_mask_single_batch() {
        let lengths = [3];
        let mut out = mask_buf(5);
        padding_mask(&lengths, 5, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_padding_mask_buffer_too_small() {
        let lengths = [2, 3];
        let mut out = mask_buf(3);
        let result = padding_mask(&lengths, 4, &mut out);
        assert!(result.is_err());
    }

    // ── Combined mask ───────────────────────────────────────────────

    #[test]
    fn test_combined_mask_full_seq() {
        // When all sequences are full length, combined == causal
        let lengths = [4];
        let mut combined = mask_buf(16);
        let mut causal = mask_buf(16);
        combined_mask(&lengths, 4, &mut combined).unwrap();
        causal_mask(4, &mut causal).unwrap();
        assert_eq!(combined, causal);
    }

    #[test]
    fn test_combined_mask_short_seq() {
        let lengths = [2];
        let seq_len = 4;
        let mut out = mask_buf(16);
        combined_mask(&lengths, seq_len, &mut out).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: col 0 valid (col<=row, col<2, row<2)
            1.0, 1.0, 0.0, 0.0, // row 1: cols 0,1 valid
            0.0, 0.0, 0.0, 0.0, // row 2: row >= valid_len → all masked
            0.0, 0.0, 0.0, 0.0, // row 3: row >= valid_len → all masked
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_combined_mask_batch() {
        let lengths = [2, 3];
        let seq_len = 3;
        let mut out = mask_buf(2 * 9);
        combined_mask(&lengths, seq_len, &mut out).unwrap();
        // Batch 0 (len=2): only first 2 rows/cols valid + causal
        assert_eq!(out[0], 1.0); // (0,0)
        assert_eq!(out[1], 0.0); // (0,1) — col > row
        assert_eq!(out[3], 1.0); // (1,0)
        assert_eq!(out[4], 1.0); // (1,1)
        assert_eq!(out[6], 0.0); // (2,0) — row >= valid_len
        // Batch 1 (len=3): full causal
        let b1 = 9;
        assert_eq!(out[b1], 1.0); // (0,0)
        assert_eq!(out[b1 + 4], 1.0); // (1,1)
        assert_eq!(out[b1 + 8], 1.0); // (2,2)
    }

    #[test]
    fn test_combined_mask_causal_subset() {
        // Combined mask should be a subset of causal mask
        let lengths = [3];
        let seq_len = 4;
        let mut combined = mask_buf(16);
        let mut causal = mask_buf(16);
        combined_mask(&lengths, seq_len, &mut combined).unwrap();
        causal_mask(seq_len, &mut causal).unwrap();
        for i in 0..16 {
            if combined[i] == 1.0 {
                assert_eq!(causal[i], 1.0, "combined mask is 1 where causal is 0 at index {i}");
            }
        }
    }

    #[test]
    fn test_combined_mask_buffer_too_small() {
        let lengths = [2];
        let mut out = mask_buf(8);
        let result = combined_mask(&lengths, 4, &mut out);
        assert!(result.is_err());
    }

    // ── Sliding window mask ─────────────────────────────────────────

    #[test]
    fn test_sliding_window_size1() {
        // Window=1 → diagonal only
        let n = 4;
        let mut out = mask_buf(n * n);
        sliding_window_mask(n, 1, &mut out).unwrap();
        for row in 0..n {
            for col in 0..n {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_eq!(out[row * n + col], expected, "mismatch at ({row}, {col})");
            }
        }
    }

    #[test]
    fn test_sliding_window_size2() {
        let n = 4;
        let mut out = mask_buf(n * n);
        sliding_window_mask(n, 2, &mut out).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: only col 0 (dist=0 < 2)
            1.0, 1.0, 0.0, 0.0, // row 1: cols 0,1 (dist 1,0 < 2)
            0.0, 1.0, 1.0, 0.0, // row 2: cols 1,2 (dist 1,0 < 2)
            0.0, 0.0, 1.0, 1.0, // row 3: cols 2,3 (dist 1,0 < 2)
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_sliding_window_size64() {
        let n = 64;
        let mut out = mask_buf(n * n);
        sliding_window_mask(n, 64, &mut out).unwrap();
        // Window >= seq_len → equivalent to causal mask
        let mut causal = mask_buf(n * n);
        causal_mask(n, &mut causal).unwrap();
        assert_eq!(out, causal);
    }

    #[test]
    fn test_sliding_window_size128_on_64() {
        let n = 64;
        let mut out = mask_buf(n * n);
        sliding_window_mask(n, 128, &mut out).unwrap();
        // Window > seq_len → still just causal
        let mut causal = mask_buf(n * n);
        causal_mask(n, &mut causal).unwrap();
        assert_eq!(out, causal);
    }

    #[test]
    fn test_sliding_window_size512() {
        let n = 512;
        let mut out = mask_buf(n * n);
        sliding_window_mask(n, 512, &mut out).unwrap();
        let mut causal = mask_buf(n * n);
        causal_mask(n, &mut causal).unwrap();
        assert_eq!(out, causal);
    }

    #[test]
    fn test_sliding_window_subset_of_causal() {
        let n = 8;
        let mut window = mask_buf(n * n);
        let mut causal = mask_buf(n * n);
        sliding_window_mask(n, 3, &mut window).unwrap();
        causal_mask(n, &mut causal).unwrap();
        for i in 0..(n * n) {
            if window[i] == 1.0 {
                assert_eq!(causal[i], 1.0, "window is 1 where causal is 0 at {i}");
            }
        }
    }

    #[test]
    fn test_sliding_window_zero_error() {
        let mut out = mask_buf(16);
        let result = sliding_window_mask(4, 0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_sliding_window_seq0() {
        let mut out = mask_buf(0);
        sliding_window_mask(0, 4, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_sliding_window_seq1() {
        let mut out = mask_buf(1);
        sliding_window_mask(1, 1, &mut out).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_sliding_window_band_count() {
        // Window of size w: row i has min(w, i+1) non-zero entries
        let n = 8;
        let w = 3;
        let mut out = mask_buf(n * n);
        sliding_window_mask(n, w, &mut out).unwrap();
        for row in 0..n {
            let row_sum: f32 = out[row * n..(row + 1) * n].iter().sum();
            let expected = (row + 1).min(w) as f32;
            assert_eq!(row_sum, expected, "row {row} count mismatch");
        }
    }

    // ── Block sparse mask ───────────────────────────────────────────

    #[test]
    fn test_block_sparse_identity_block1() {
        // block_size=1 → diagonal only
        let n = 4;
        let mut out = mask_buf(n * n);
        block_sparse_mask(n, 1, &mut out).unwrap();
        for row in 0..n {
            for col in 0..n {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert_eq!(out[row * n + col], expected, "mismatch at ({row}, {col})");
            }
        }
    }

    #[test]
    fn test_block_sparse_full_block() {
        // block_size >= seq_len → full mask (all ones)
        let n = 4;
        let mut out = mask_buf(n * n);
        block_sparse_mask(n, n, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_block_sparse_2x2() {
        let n = 4;
        let mut out = mask_buf(n * n);
        block_sparse_mask(n, 2, &mut out).unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_block_sparse_symmetry() {
        let n = 8;
        let mut out = mask_buf(n * n);
        block_sparse_mask(n, 4, &mut out).unwrap();
        for row in 0..n {
            for col in 0..n {
                assert_eq!(
                    out[row * n + col],
                    out[col * n + row],
                    "not symmetric at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn test_block_sparse_zero_error() {
        let mut out = mask_buf(16);
        let result = block_sparse_mask(4, 0, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_sparse_seq0() {
        let mut out = mask_buf(0);
        block_sparse_mask(0, 2, &mut out).unwrap();
        assert!(out.is_empty());
    }

    // ── ALiBi mask ──────────────────────────────────────────────────

    #[test]
    fn test_alibi_slopes_power_of_2() {
        let slopes = compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        // First slope should be 2^(-1) = 0.5
        assert!((slopes[0] - 0.5).abs() < 1e-6, "first slope: {}", slopes[0]);
        // Slopes should be decreasing
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1], "slopes not decreasing at {i}");
        }
    }

    #[test]
    fn test_alibi_slopes_1_head() {
        let slopes = compute_alibi_slopes(1);
        assert_eq!(slopes.len(), 1);
        // 2^(-8/1) = 2^(-8) ≈ 0.00390625
        assert!((slopes[0] - 0.003906_25).abs() < 1e-6, "slope: {}", slopes[0]);
    }

    #[test]
    fn test_alibi_slopes_32_heads() {
        let slopes = compute_alibi_slopes(32);
        assert_eq!(slopes.len(), 32);
        // All slopes positive
        assert!(slopes.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_alibi_slopes_non_power_of_2() {
        let slopes = compute_alibi_slopes(6);
        assert_eq!(slopes.len(), 6);
        assert!(slopes.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_alibi_slopes_0_heads() {
        let slopes = compute_alibi_slopes(0);
        assert!(slopes.is_empty());
    }

    #[test]
    fn test_alibi_mask_diagonal_zero() {
        let slopes = compute_alibi_slopes(2);
        let n = 4;
        let mut out = mask_buf(2 * n * n);
        alibi_mask(n, &slopes, &mut out).unwrap();
        // Diagonal entries: distance=0 → bias=0
        for h in 0..2 {
            for i in 0..n {
                assert_eq!(
                    out[h * n * n + i * n + i],
                    0.0,
                    "diagonal not zero for head {h} pos {i}"
                );
            }
        }
    }

    #[test]
    fn test_alibi_mask_symmetry() {
        let slopes = compute_alibi_slopes(2);
        let n = 4;
        let mut out = mask_buf(2 * n * n);
        alibi_mask(n, &slopes, &mut out).unwrap();
        // |i-j| is symmetric → bias is symmetric
        for h in 0..2 {
            let h_off = h * n * n;
            for row in 0..n {
                for col in 0..n {
                    assert_eq!(
                        out[h_off + row * n + col],
                        out[h_off + col * n + row],
                        "not symmetric at head {h} ({row},{col})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_alibi_mask_negative_bias() {
        let slopes = compute_alibi_slopes(2);
        let n = 4;
        let mut out = mask_buf(2 * n * n);
        alibi_mask(n, &slopes, &mut out).unwrap();
        // Off-diagonal entries should be negative (−slope * distance)
        for h in 0..2 {
            let h_off = h * n * n;
            for row in 0..n {
                for col in 0..n {
                    if row != col {
                        assert!(
                            out[h_off + row * n + col] < 0.0,
                            "expected negative at head {h} ({row},{col})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_alibi_mask_larger_distance_more_negative() {
        let slopes = compute_alibi_slopes(1);
        let n = 8;
        let mut out = mask_buf(n * n);
        alibi_mask(n, &slopes, &mut out).unwrap();
        // For row 7: bias at col 0 should be more negative than col 6
        let bias_far = out[7 * n + 0]; // distance=7
        let bias_near = out[7 * n + 6]; // distance=1
        assert!(bias_far < bias_near, "far={bias_far} should be < near={bias_near}");
    }

    #[test]
    fn test_alibi_mask_seq0() {
        let slopes = compute_alibi_slopes(2);
        let mut out = mask_buf(0);
        alibi_mask(0, &slopes, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_alibi_mask_buffer_too_small() {
        let slopes = compute_alibi_slopes(2);
        let mut out = mask_buf(4);
        let result = alibi_mask(4, &slopes, &mut out);
        assert!(result.is_err());
    }

    // ── Mask application ────────────────────────────────────────────

    #[test]
    fn test_apply_mask_to_scores_basic() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        apply_mask_to_scores(&mut scores, &mask).unwrap();
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], NEG_INF);
        assert_eq!(scores[2], 3.0);
        assert_eq!(scores[3], NEG_INF);
    }

    #[test]
    fn test_apply_mask_to_scores_all_unmasked() {
        let original = vec![1.0, 2.0, 3.0];
        let mut scores = original.clone();
        let mask = vec![1.0, 1.0, 1.0];
        apply_mask_to_scores(&mut scores, &mask).unwrap();
        assert_eq!(scores, original);
    }

    #[test]
    fn test_apply_mask_to_scores_all_masked() {
        let mut scores = vec![1.0, 2.0, 3.0];
        let mask = vec![0.0, 0.0, 0.0];
        apply_mask_to_scores(&mut scores, &mask).unwrap();
        assert!(scores.iter().all(|&s| s == NEG_INF));
    }

    #[test]
    fn test_apply_mask_to_scores_length_mismatch() {
        let mut scores = vec![1.0, 2.0];
        let mask = [1.0];
        let result = apply_mask_to_scores(&mut scores, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_mask_additive_basic() {
        let mut scores = vec![0.5, -0.3, 1.2, 0.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        apply_mask_additive(&mut scores, &mask).unwrap();
        assert_eq!(scores[0], 0.5);
        assert_eq!(scores[1], NEG_INF);
        assert_eq!(scores[2], 1.2);
        assert_eq!(scores[3], NEG_INF);
    }

    #[test]
    fn test_apply_mask_additive_length_mismatch() {
        let mut scores = [1.0];
        let mask = vec![1.0, 0.0];
        let result = apply_mask_additive(&mut scores, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_mask_multiplicative_basic() {
        let mut scores = vec![2.0, 3.0, 4.0, 5.0];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        apply_mask_multiplicative(&mut scores, &mask).unwrap();
        assert_eq!(scores, vec![2.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn test_apply_mask_multiplicative_all_ones() {
        let original = vec![2.0, 3.0, 4.0];
        let mut scores = original.clone();
        let mask = vec![1.0, 1.0, 1.0];
        apply_mask_multiplicative(&mut scores, &mask).unwrap();
        assert_eq!(scores, original);
    }

    #[test]
    fn test_apply_mask_multiplicative_all_zeros() {
        let mut scores = vec![2.0, 3.0, 4.0];
        let mask = vec![0.0, 0.0, 0.0];
        apply_mask_multiplicative(&mut scores, &mask).unwrap();
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_apply_mask_multiplicative_length_mismatch() {
        let mut scores = vec![1.0, 2.0, 3.0];
        let mask = [1.0];
        let result = apply_mask_multiplicative(&mut scores, &mask);
        assert!(result.is_err());
    }

    // ── Prefix mask ─────────────────────────────────────────────────

    #[test]
    fn test_prefix_mask_full_prefix() {
        // prefix_len == seq_len → full bidirectional (all ones)
        let n = 4;
        let mut out = mask_buf(n * n);
        create_prefix_mask(n, n, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_prefix_mask_zero_prefix() {
        // prefix_len == 0 → pure causal
        let n = 4;
        let mut prefix = mask_buf(n * n);
        let mut causal = mask_buf(n * n);
        create_prefix_mask(n, 0, &mut prefix).unwrap();
        causal_mask(n, &mut causal).unwrap();
        assert_eq!(prefix, causal);
    }

    #[test]
    fn test_prefix_mask_partial() {
        let n = 4;
        let prefix_len = 2;
        let mut out = mask_buf(n * n);
        create_prefix_mask(n, prefix_len, &mut out).unwrap();
        // Rows 0,1 (prefix): all 1s
        for row in 0..prefix_len {
            for col in 0..n {
                assert_eq!(out[row * n + col], 1.0, "prefix row {row} col {col} should be 1");
            }
        }
        // Rows 2,3 (suffix): causal
        for row in prefix_len..n {
            for col in 0..n {
                let expected = if col <= row { 1.0 } else { 0.0 };
                assert_eq!(out[row * n + col], expected, "suffix row {row} col {col}");
            }
        }
    }

    #[test]
    fn test_prefix_mask_seq1_prefix1() {
        let mut out = mask_buf(1);
        create_prefix_mask(1, 1, &mut out).unwrap();
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn test_prefix_mask_seq1_prefix0() {
        let mut out = mask_buf(1);
        create_prefix_mask(1, 0, &mut out).unwrap();
        assert_eq!(out, vec![1.0]); // causal: col 0 <= row 0
    }

    #[test]
    fn test_prefix_mask_exceeds_seq_len() {
        let mut out = mask_buf(16);
        let result = create_prefix_mask(4, 5, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_prefix_mask_buffer_too_small() {
        let mut out = mask_buf(8);
        let result = create_prefix_mask(4, 2, &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn test_prefix_mask_superset_of_causal() {
        // Prefix mask should be superset of causal mask
        let n = 8;
        let mut prefix = mask_buf(n * n);
        let mut causal = mask_buf(n * n);
        create_prefix_mask(n, 4, &mut prefix).unwrap();
        causal_mask(n, &mut causal).unwrap();
        for i in 0..(n * n) {
            if causal[i] == 1.0 {
                assert_eq!(prefix[i], 1.0, "prefix should be 1 where causal is 1 at index {i}");
            }
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_causal_mask_empty_then_seq1() {
        let mut out0 = mask_buf(0);
        causal_mask(0, &mut out0).unwrap();
        assert!(out0.is_empty());

        let mut out1 = mask_buf(1);
        causal_mask(1, &mut out1).unwrap();
        assert_eq!(out1, vec![1.0]);
    }

    #[test]
    fn test_neg_inf_not_nan() {
        assert!(!NEG_INF.is_nan());
        assert!(NEG_INF.is_finite());
        assert!(NEG_INF < 0.0);
    }

    #[test]
    fn test_masked_scores_neg_inf_value() {
        let mut scores = [1.0; 4];
        let mask = vec![1.0, 0.0, 1.0, 0.0];
        apply_mask_to_scores(&mut scores, &mask).unwrap();
        assert_eq!(scores[1], NEG_INF);
        assert_eq!(scores[3], NEG_INF);
        assert_eq!(scores[0], 1.0);
    }

    #[test]
    fn test_multiplicative_mask_zeros_exact() {
        let mut scores = vec![42.0, -7.5, 100.0];
        let mask = vec![0.0, 0.0, 0.0];
        apply_mask_multiplicative(&mut scores, &mask).unwrap();
        for &s in &scores {
            assert_eq!(s, 0.0, "expected exact zero, got {s}");
        }
    }

    // ── Config types ────────────────────────────────────────────────

    #[test]
    fn test_attention_mask_config_grid_dim() {
        let cfg = AttentionMaskConfig::new(64);
        let (gx, gy, gz) = cfg.grid_dim();
        assert!(gx > 0);
        assert_eq!(gy, 1);
        assert_eq!(gz, 1);
    }

    #[test]
    fn test_sliding_window_config_zero_err() {
        let result = SlidingWindowConfig::new(64, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_block_sparse_config_zero_err() {
        let result = BlockSparseConfig::new(64, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_alibi_config_zero_heads_err() {
        let result = AlibiConfig::new(64, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_prefix_mask_config_exceeds_err() {
        let result = PrefixMaskConfig::new(4, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_prefix_mask_config_valid() {
        let cfg = PrefixMaskConfig::new(8, 4).unwrap();
        assert_eq!(cfg.seq_len, 8);
        assert_eq!(cfg.prefix_len, 4);
    }

    // ── Print helper guard (unused but covers _print_mask) ──────────

    #[test]
    fn test_print_mask_helper() {
        // Just ensure the helper doesn't panic
        let mask = vec![1.0, 0.0, 1.0, 1.0];
        print_mask(&mask, 2, 2);
    }

    // ── CUDA kernel source presence ─────────────────────────────────

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    #[test]
    fn test_kernel_source_contains_functions() {
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("causal_mask_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("padding_mask_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("sliding_window_mask_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("block_sparse_mask_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("alibi_mask_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("prefix_mask_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("apply_mask_additive_f32"));
        assert!(ATTENTION_MASK_KERNEL_SRC.contains("apply_mask_multiplicative_f32"));
    }
}
