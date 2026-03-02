//! Flash decoding CUDA kernel for efficient autoregressive inference.
//!
//! # Kernel strategy
//!
//! During autoregressive generation each step produces a **single query token**
//! that must attend to the entire KV cache.  Standard attention is bottlenecked
//! by memory bandwidth because a single query row cannot saturate the GPU's
//! compute units.  Flash decoding solves this by:
//!
//! 1. **Splitting** the KV cache across multiple thread-blocks (`num_splits`).
//! 2. Each block computes a **partial softmax** (local max + exp-sum + weighted
//!    V accumulator) over its assigned KV range.
//! 3. A lightweight **merge** kernel combines the partial results using the
//!    log-sum-exp correction trick, yielding the exact same output as
//!    single-pass attention.
//!
//! This two-phase approach increases parallelism from `O(num_heads)` blocks to
//! `O(num_heads × num_splits)`, improving GPU utilisation on long sequences.
//!
//! # Variants
//!
//! - [`flash_decode_attention`] — standard multi-head flash decoding
//! - [`flash_decode_with_alibi`] — with ALiBi (Attention with Linear Biases)
//! - [`flash_decode_gqa`] — grouped-query attention (fewer KV heads)
//! - [`paged_flash_decode`] — paged KV cache (non-contiguous blocks)
//!
//! # CPU fallback
//!
//! All functions have pure-Rust CPU implementations for correctness testing
//! and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

/// Inline CUDA C source for the flash decoding kernels.
///
/// Two kernels:
/// - `flash_decode_partial_f32`: per-split partial attention
/// - `flash_decode_merge_f32`: merge partial outputs via log-sum-exp
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub const FLASH_DECODE_KERNEL_SRC: &str = r#"
extern "C" __global__ void flash_decode_partial_f32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_out,
    float* __restrict__ partial_lse,
    int seq_len_kv,
    int head_dim,
    int num_splits,
    float scale)
{
    int head_idx = blockIdx.y;
    int split_idx = blockIdx.x;

    int split_size = (seq_len_kv + num_splits - 1) / num_splits;
    int kv_start = split_idx * split_size;
    int kv_end = kv_start + split_size;
    if (kv_end > seq_len_kv) kv_end = seq_len_kv;
    if (kv_start >= seq_len_kv) return;

    const float* q_row = Q + head_idx * head_dim;
    float local_max = -1e30f;

    // Pass 1: compute scores and find local max
    for (int j = kv_start; j < kv_end; j++) {
        const float* k_row = K + (head_idx * seq_len_kv + j) * head_dim;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q_row[d] * k_row[d];
        }
        // Warp reduction for dot product would go here in production
        dot *= scale;
        if (dot > local_max) local_max = dot;
    }

    // Pass 2: exp-sum and weighted V
    float exp_sum = 0.0f;
    int out_offset = (head_idx * num_splits + split_idx) * head_dim;
    for (int d = 0; d < head_dim; d++) {
        partial_out[out_offset + d] = 0.0f;
    }

    for (int j = kv_start; j < kv_end; j++) {
        const float* k_row = K + (head_idx * seq_len_kv + j) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_row[d] * k_row[d];
        }
        dot *= scale;
        float w = expf(dot - local_max);
        exp_sum += w;
        const float* v_row = V + (head_idx * seq_len_kv + j) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            partial_out[out_offset + d] += w * v_row[d];
        }
    }

    // Store log-sum-exp = local_max + log(exp_sum)
    int lse_idx = head_idx * num_splits + split_idx;
    partial_lse[lse_idx] = local_max + logf(exp_sum + 1e-10f);
}

extern "C" __global__ void flash_decode_merge_f32(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    float* __restrict__ output,
    int head_dim,
    int num_splits)
{
    int head_idx = blockIdx.x;

    // Find global max of partial LSEs
    float global_max = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[head_idx * num_splits + s];
        if (lse > global_max) global_max = lse;
    }

    // Merge with correction
    float total_weight = 0.0f;
    int o_offset = head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
        output[o_offset + d] = 0.0f;
    }

    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[head_idx * num_splits + s];
        float correction = expf(lse - global_max);
        total_weight += correction;
        int p_offset = (head_idx * num_splits + s) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            output[o_offset + d] += partial_out[p_offset + d] * correction;
        }
    }

    // Normalise
    if (total_weight > 0.0f) {
        float inv = 1.0f / total_weight;
        for (int d = 0; d < head_dim; d++) {
            output[o_offset + d] *= inv;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for flash decoding.
#[derive(Debug, Clone)]
pub struct FlashDecodingConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head embedding dimension (typically 32, 64, or 128).
    pub head_dim: usize,
    /// Number of KV-cache positions to attend to.
    pub max_seq_len: usize,
    /// Number of splits for parallelising the KV range.
    pub num_splits: usize,
    /// Softmax temperature scale (`1.0 / sqrt(head_dim)` by default).
    pub scale: f32,
}

impl FlashDecodingConfig {
    /// Create a new flash decoding configuration.
    ///
    /// `num_splits` is auto-tuned to `ceil(max_seq_len / 256)` clamped to
    /// `[1, 128]` when passed as `0`.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        num_splits: usize,
    ) -> Result<Self> {
        if num_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "FlashDecodingConfig: num_heads must be non-zero".into(),
            }
            .into());
        }
        if head_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "FlashDecodingConfig: head_dim must be non-zero".into(),
            }
            .into());
        }
        if max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "FlashDecodingConfig: max_seq_len must be non-zero".into(),
            }
            .into());
        }

        let num_splits =
            if num_splits == 0 { (max_seq_len.div_ceil(256)).clamp(1, 128) } else { num_splits };

        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self { num_heads, head_dim, max_seq_len, num_splits, scale })
    }

    /// Override the default softmax scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Compute CUDA grid dimensions for the partial kernel.
    ///
    /// `(num_splits, num_heads, 1)`
    pub fn partial_grid_dim(&self) -> (u32, u32, u32) {
        (self.num_splits as u32, self.num_heads as u32, 1)
    }

    /// Compute CUDA grid dimensions for the merge kernel.
    ///
    /// `(num_heads, 1, 1)`
    pub fn merge_grid_dim(&self) -> (u32, u32, u32) {
        (self.num_heads as u32, 1, 1)
    }

    /// KV positions assigned to a single split.
    pub fn split_size(&self) -> usize {
        self.max_seq_len.div_ceil(self.num_splits)
    }
}

// ---------------------------------------------------------------------------
// Paged KV cache descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a paged KV cache.
///
/// Instead of a single contiguous buffer the cache is composed of fixed-size
/// pages that can be allocated and freed independently (similar to OS virtual
/// memory).
#[derive(Debug, Clone)]
pub struct PagedKvDescriptor {
    /// Number of tokens per page.
    pub page_size: usize,
    /// Logical page table: `page_table[i]` is the physical page index for
    /// logical page `i`.  Length = `ceil(max_seq_len / page_size)`.
    pub page_table: Vec<usize>,
    /// Total number of physical pages allocated.
    pub num_physical_pages: usize,
}

// ---------------------------------------------------------------------------
// GQA configuration
// ---------------------------------------------------------------------------

/// Configuration extension for grouped-query attention.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of KV heads (must divide `num_q_heads` evenly).
    pub num_kv_heads: usize,
}

impl GqaConfig {
    /// Create a new GQA config, validating the group ratio.
    pub fn new(num_q_heads: usize, num_kv_heads: usize) -> Result<Self> {
        if num_q_heads == 0 || num_kv_heads == 0 {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "GqaConfig: heads must be non-zero: q={num_q_heads}, kv={num_kv_heads}"
                ),
            }
            .into());
        }
        if !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "GqaConfig: num_q_heads ({num_q_heads}) must be divisible \
                     by num_kv_heads ({num_kv_heads})"
                ),
            }
            .into());
        }
        Ok(Self { num_q_heads, num_kv_heads })
    }

    /// Number of query heads sharing each KV head.
    pub fn group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

// ---------------------------------------------------------------------------
// CUDA launch stubs
// ---------------------------------------------------------------------------

/// Launch the flash decoding CUDA kernel (partial + merge).
///
/// # Errors
///
/// Returns `KernelError::GpuError` — scaffold only.
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub fn launch_flash_decode(
    _q: &[f32],
    _k: &[f32],
    _v: &[f32],
    _output: &mut [f32],
    config: &FlashDecodingConfig,
) -> Result<()> {
    log::debug!(
        "flash_decode stub: heads={}, dim={}, seq={}, splits={}, grid_partial={:?}",
        config.num_heads,
        config.head_dim,
        config.max_seq_len,
        config.num_splits,
        config.partial_grid_dim(),
    );
    Err(KernelError::GpuError {
        reason: "Flash decoding CUDA kernel not yet compiled — scaffold only".into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// CPU helpers
// ---------------------------------------------------------------------------

/// Numerically stable softmax in-place.
fn softmax_inplace(scores: &mut [f32]) {
    let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for s in scores.iter_mut() {
        let e = (*s - row_max).exp();
        *s = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// Reference single-head, single-query attention (naive).
///
/// Computes `softmax(q · Kᵀ · scale) · V` for a single query vector against
/// the full KV cache.  Used as the ground-truth baseline for testing.
#[cfg(test)]
fn naive_single_query_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0_f32; seq_len];
    for j in 0..seq_len {
        let mut dot = 0.0_f32;
        for d in 0..head_dim {
            dot += q[d] * k[j * head_dim + d];
        }
        scores[j] = dot * scale;
    }
    softmax_inplace(&mut scores);

    let mut out = vec![0.0_f32; head_dim];
    for j in 0..seq_len {
        for d in 0..head_dim {
            out[d] += scores[j] * v[j * head_dim + d];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CPU fallback: split KV across blocks
// ---------------------------------------------------------------------------

/// Split the KV range `[0, seq_len)` into `num_splits` contiguous sub-ranges.
///
/// Returns a vector of `(start, end)` pairs.
pub fn split_kv_across_blocks(seq_len: usize, num_splits: usize) -> Vec<(usize, usize)> {
    if num_splits == 0 || seq_len == 0 {
        return vec![];
    }
    let split_size = seq_len.div_ceil(num_splits);
    let mut ranges = Vec::with_capacity(num_splits);
    let mut start = 0;
    while start < seq_len {
        let end = (start + split_size).min(seq_len);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

// ---------------------------------------------------------------------------
// CPU fallback: partial softmax
// ---------------------------------------------------------------------------

/// Compute partial softmax over a KV sub-range for a single query vector.
///
/// Returns `(normalised_v, log_sum_exp)` where:
/// - `normalised_v` is the attention-weighted sum of V rows for this split,
///   normalised within the split: `softmax(scores) · V`
/// - `log_sum_exp = max + ln(Σ exp(score_j − max))` — encodes the total
///   probability mass in this split for correct cross-split merging
pub fn partial_softmax(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_start: usize,
    kv_end: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, f32) {
    let len = kv_end - kv_start;
    if len == 0 {
        return (vec![0.0; head_dim], f32::NEG_INFINITY);
    }

    // Compute scores
    let mut scores = vec![0.0_f32; len];
    let mut local_max = f32::NEG_INFINITY;
    for (ci, j) in (kv_start..kv_end).enumerate() {
        let mut dot = 0.0_f32;
        for d in 0..head_dim {
            dot += q[d] * k[j * head_dim + d];
        }
        scores[ci] = dot * scale;
        if scores[ci] > local_max {
            local_max = scores[ci];
        }
    }

    // Exp-sum and weighted V
    let mut exp_sum = 0.0_f32;
    let mut weighted_v = vec![0.0_f32; head_dim];
    for (ci, j) in (kv_start..kv_end).enumerate() {
        let w = (scores[ci] - local_max).exp();
        exp_sum += w;
        for d in 0..head_dim {
            weighted_v[d] += w * v[j * head_dim + d];
        }
    }

    // Normalise within this split so merge_partial_attention can use LSE weighting
    if exp_sum > 0.0 {
        let inv = 1.0 / exp_sum;
        for wv in weighted_v.iter_mut() {
            *wv *= inv;
        }
    }

    let lse = local_max + (exp_sum + 1e-10).ln();
    (weighted_v, lse)
}

// ---------------------------------------------------------------------------
// CPU fallback: merge partial attention
// ---------------------------------------------------------------------------

/// Merge partial attention outputs using log-sum-exp correction.
///
/// Given `N` partial results `(weighted_v_i, lse_i)` from [`partial_softmax`],
/// produces the correctly normalised attention output.
pub fn merge_partial_attention(
    partial_outputs: &[Vec<f32>],
    partial_lses: &[f32],
    head_dim: usize,
) -> Vec<f32> {
    let n = partial_lses.len();
    if n == 0 {
        return vec![0.0; head_dim];
    }

    let global_max = partial_lses.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut merged = vec![0.0_f32; head_dim];
    let mut total_weight = 0.0_f32;

    for i in 0..n {
        let correction = (partial_lses[i] - global_max).exp();
        total_weight += correction;
        for d in 0..head_dim {
            merged[d] += partial_outputs[i][d] * correction;
        }
    }

    if total_weight > 0.0 {
        let inv = 1.0 / total_weight;
        for m in merged.iter_mut() {
            *m *= inv;
        }
    }

    merged
}

// ---------------------------------------------------------------------------
// CPU fallback: flash_decode_attention
// ---------------------------------------------------------------------------

/// Flash decoding for single-token generation (CPU fallback).
///
/// # Layout
///
/// * `q` — `[num_heads, head_dim]`
/// * `k` — `[num_heads, seq_len, head_dim]`
/// * `v` — `[num_heads, seq_len, head_dim]`
///
/// # Returns
///
/// Output `[num_heads, head_dim]` as a flat `Vec<f32>`.
pub fn flash_decode_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &FlashDecodingConfig,
) -> Result<Vec<f32>> {
    validate_flash_inputs(q, k, v, config)?;

    let h = config.num_heads;
    let d = config.head_dim;
    let seq = config.max_seq_len;
    let splits = split_kv_across_blocks(seq, config.num_splits);

    let mut output = vec![0.0_f32; h * d];

    for head in 0..h {
        let q_head = &q[head * d..(head + 1) * d];
        let k_head = &k[head * seq * d..(head + 1) * seq * d];
        let v_head = &v[head * seq * d..(head + 1) * seq * d];

        let mut partial_outs = Vec::with_capacity(splits.len());
        let mut partial_lses = Vec::with_capacity(splits.len());

        for &(start, end) in &splits {
            let (wv, lse) = partial_softmax(q_head, k_head, v_head, start, end, d, config.scale);
            partial_outs.push(wv);
            partial_lses.push(lse);
        }

        let merged = merge_partial_attention(&partial_outs, &partial_lses, d);
        output[head * d..(head + 1) * d].copy_from_slice(&merged);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback: flash_decode_with_alibi
// ---------------------------------------------------------------------------

/// Flash decoding with ALiBi (Attention with Linear Biases) position encoding.
///
/// Each head `h` applies a per-position bias of `slope_h * (position - seq_len + 1)`
/// to the attention scores, where `slope_h = 2^(-8h/num_heads)`.
///
/// # Layout
///
/// Same as [`flash_decode_attention`].
pub fn flash_decode_with_alibi(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &FlashDecodingConfig,
) -> Result<Vec<f32>> {
    validate_flash_inputs(q, k, v, config)?;

    let h = config.num_heads;
    let d = config.head_dim;
    let seq = config.max_seq_len;

    let mut output = vec![0.0_f32; h * d];

    for head in 0..h {
        let slope = alibi_slope(head, h);
        let q_head = &q[head * d..(head + 1) * d];
        let k_head = &k[head * seq * d..(head + 1) * seq * d];
        let v_head = &v[head * seq * d..(head + 1) * seq * d];

        // Compute scores with ALiBi bias
        let mut scores = vec![0.0_f32; seq];
        let mut local_max = f32::NEG_INFINITY;
        for j in 0..seq {
            let mut dot = 0.0_f32;
            for dd in 0..d {
                dot += q_head[dd] * k_head[j * d + dd];
            }
            // ALiBi: bias = slope * (j - seq + 1), so position 0 gets the
            // most negative bias and the last position gets 0.
            let bias = slope * (j as f32 - seq as f32 + 1.0);
            scores[j] = dot * config.scale + bias;
            if scores[j] > local_max {
                local_max = scores[j];
            }
        }

        softmax_inplace(&mut scores);

        for dd in 0..d {
            let mut acc = 0.0_f32;
            for j in 0..seq {
                acc += scores[j] * v_head[j * d + dd];
            }
            output[head * d + dd] = acc;
        }
    }

    Ok(output)
}

/// Compute the ALiBi slope for head `h` out of `num_heads` total.
///
/// `slope = 2^(-8 * (h + 1) / num_heads)`
fn alibi_slope(h: usize, num_heads: usize) -> f32 {
    2.0_f32.powf(-8.0 * (h as f32 + 1.0) / num_heads as f32)
}

// ---------------------------------------------------------------------------
// CPU fallback: flash_decode_gqa
// ---------------------------------------------------------------------------

/// Flash decoding for grouped-query attention (CPU fallback).
///
/// Multiple query heads share the same KV head.
///
/// # Layout
///
/// * `q` — `[num_q_heads, head_dim]`
/// * `k` — `[num_kv_heads, seq_len, head_dim]`
/// * `v` — `[num_kv_heads, seq_len, head_dim]`
pub fn flash_decode_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &FlashDecodingConfig,
    gqa: &GqaConfig,
) -> Result<Vec<f32>> {
    let d = config.head_dim;
    let seq = config.max_seq_len;

    let q_expected = gqa.num_q_heads * d;
    let kv_expected = gqa.num_kv_heads * seq * d;
    if q.len() < q_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("flash_decode_gqa: q length {}, expected {q_expected}", q.len()),
        }
        .into());
    }
    if k.len() < kv_expected || v.len() < kv_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "flash_decode_gqa: k/v length mismatch, expected {kv_expected}, \
                 got k={}, v={}",
                k.len(),
                v.len()
            ),
        }
        .into());
    }

    let group = gqa.group_size();
    let splits = split_kv_across_blocks(seq, config.num_splits);

    let mut output = vec![0.0_f32; gqa.num_q_heads * d];

    for q_head in 0..gqa.num_q_heads {
        let kv_head = q_head / group;
        let q_slice = &q[q_head * d..(q_head + 1) * d];
        let k_slice = &k[kv_head * seq * d..(kv_head + 1) * seq * d];
        let v_slice = &v[kv_head * seq * d..(kv_head + 1) * seq * d];

        let mut partial_outs = Vec::with_capacity(splits.len());
        let mut partial_lses = Vec::with_capacity(splits.len());

        for &(start, end) in &splits {
            let (wv, lse) = partial_softmax(q_slice, k_slice, v_slice, start, end, d, config.scale);
            partial_outs.push(wv);
            partial_lses.push(lse);
        }

        let merged = merge_partial_attention(&partial_outs, &partial_lses, d);
        output[q_head * d..(q_head + 1) * d].copy_from_slice(&merged);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// CPU fallback: paged_flash_decode
// ---------------------------------------------------------------------------

/// Flash decoding with paged KV cache (CPU fallback).
///
/// K and V are stored in non-contiguous pages.  The `page_table` maps logical
/// page indices to physical page indices within the flat `k_pages`/`v_pages`
/// buffers.
///
/// # Layout
///
/// * `q`        — `[num_heads, head_dim]`
/// * `k_pages`  — `[num_physical_pages, page_size, head_dim]` (all heads interleaved)
/// * `v_pages`  — same layout as `k_pages`
/// * `paged`    — page descriptor
///
/// Within each physical page the layout is `[page_size, head_dim]` for a
/// single head.  For multi-head the pages are stored per-head:
/// `k_pages[head * num_physical_pages * page_size * head_dim + page * page_size * head_dim + pos * head_dim + d]`.
pub fn paged_flash_decode(
    q: &[f32],
    k_pages: &[f32],
    v_pages: &[f32],
    config: &FlashDecodingConfig,
    paged: &PagedKvDescriptor,
) -> Result<Vec<f32>> {
    let d = config.head_dim;
    let h = config.num_heads;
    let seq = config.max_seq_len;
    let ps = paged.page_size;

    if ps == 0 {
        return Err(KernelError::InvalidArguments {
            reason: "paged_flash_decode: page_size must be non-zero".into(),
        }
        .into());
    }

    let q_expected = h * d;
    if q.len() < q_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("paged_flash_decode: q length {}, expected {q_expected}", q.len()),
        }
        .into());
    }

    let num_logical_pages = seq.div_ceil(ps);
    if paged.page_table.len() < num_logical_pages {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "paged_flash_decode: page_table length {}, need {num_logical_pages}",
                paged.page_table.len()
            ),
        }
        .into());
    }

    let page_elems = ps * d;
    let per_head_pages = paged.num_physical_pages * page_elems;
    let k_expected = h * per_head_pages;
    if k_pages.len() < k_expected || v_pages.len() < k_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "paged_flash_decode: k/v page buffer length mismatch, expected {k_expected}, \
                 got k={}, v={}",
                k_pages.len(),
                v_pages.len()
            ),
        }
        .into());
    }

    let mut output = vec![0.0_f32; h * d];

    for head in 0..h {
        let q_head = &q[head * d..(head + 1) * d];
        let head_page_offset = head * per_head_pages;

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0_f32;
        let mut acc = vec![0.0_f32; d];

        let mut pos = 0usize;
        for lp in 0..num_logical_pages {
            let pp = paged.page_table[lp];
            let page_start = head_page_offset + pp * page_elems;
            let tokens_in_page = ps.min(seq - pos);

            for t in 0..tokens_in_page {
                let k_offset = page_start + t * d;
                let mut dot = 0.0_f32;
                for dd in 0..d {
                    dot += q_head[dd] * k_pages[k_offset + dd];
                }
                let score = dot * config.scale;

                // Online softmax update
                let new_max = running_max.max(score);
                if running_sum > 0.0 {
                    let correction = (running_max - new_max).exp();
                    running_sum *= correction;
                    for a in acc.iter_mut() {
                        *a *= correction;
                    }
                }
                running_max = new_max;

                let w = (score - new_max).exp();
                running_sum += w;

                let v_offset = page_start + t * d;
                for dd in 0..d {
                    acc[dd] += w * v_pages[v_offset + dd];
                }
            }

            pos += tokens_in_page;
        }

        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            for dd in 0..d {
                output[head * d + dd] = acc[dd] * inv;
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

fn validate_flash_inputs(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &FlashDecodingConfig,
) -> Result<()> {
    let q_expected = config.num_heads * config.head_dim;
    let kv_expected = config.num_heads * config.max_seq_len * config.head_dim;

    if q.len() < q_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("flash_decode: q length {}, expected {q_expected}", q.len()),
        }
        .into());
    }
    if k.len() < kv_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("flash_decode: k length {}, expected {kv_expected}", k.len()),
        }
        .into());
    }
    if v.len() < kv_expected {
        return Err(KernelError::InvalidArguments {
            reason: format!("flash_decode: v length {}, expected {kv_expected}", v.len()),
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

    // ── Helpers ───────────────────────────────────────────────────────

    /// Simple pseudo-random f32 in `[-1, 1]` from a seed (xorshift32).
    fn pseudo_rand(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        (*seed as f32) / u32::MAX as f32 * 2.0 - 1.0
    }

    fn rand_vec(len: usize, seed: &mut u32) -> Vec<f32> {
        (0..len).map(|_| pseudo_rand(seed)).collect()
    }

    /// Maximum absolute difference.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    /// Assert two slices are close within tolerance.
    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        let diff = max_abs_diff(a, b);
        assert!(
            diff < tol,
            "{msg}: max_abs_diff = {diff} >= {tol}\n  a[..8] = {:?}\n  b[..8] = {:?}",
            &a[..a.len().min(8)],
            &b[..b.len().min(8)],
        );
    }

    /// Compute naive multi-head single-query attention as ground truth.
    fn naive_multi_head(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0_f32; num_heads * head_dim];
        for h in 0..num_heads {
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let k_h = &k[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
            let v_h = &v[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
            let head_out = naive_single_query_attention(q_h, k_h, v_h, seq_len, head_dim, scale);
            out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&head_out);
        }
        out
    }

    // ── FlashDecodingConfig tests ────────────────────────────────────

    #[test]
    fn test_config_new_basic() {
        let cfg = FlashDecodingConfig::new(8, 64, 512, 4).unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.max_seq_len, 512);
        assert_eq!(cfg.num_splits, 4);
        assert!((cfg.scale - 1.0 / 8.0).abs() < 1e-6); // 1/sqrt(64)
    }

    #[test]
    fn test_config_auto_splits() {
        let cfg = FlashDecodingConfig::new(1, 64, 1024, 0).unwrap();
        assert_eq!(cfg.num_splits, 4); // ceil(1024/256)
    }

    #[test]
    fn test_config_auto_splits_small() {
        let cfg = FlashDecodingConfig::new(1, 64, 100, 0).unwrap();
        assert_eq!(cfg.num_splits, 1); // ceil(100/256) = 1
    }

    #[test]
    fn test_config_auto_splits_large() {
        // 128*256 = 32768 → clamped to 128
        let cfg = FlashDecodingConfig::new(1, 64, 40_000, 0).unwrap();
        assert_eq!(cfg.num_splits, 128);
    }

    #[test]
    fn test_config_rejects_zero_heads() {
        assert!(FlashDecodingConfig::new(0, 64, 512, 4).is_err());
    }

    #[test]
    fn test_config_rejects_zero_dim() {
        assert!(FlashDecodingConfig::new(8, 0, 512, 4).is_err());
    }

    #[test]
    fn test_config_rejects_zero_seq() {
        assert!(FlashDecodingConfig::new(8, 64, 0, 4).is_err());
    }

    #[test]
    fn test_config_custom_scale() {
        let cfg = FlashDecodingConfig::new(1, 64, 32, 1).unwrap().with_scale(0.25);
        assert!((cfg.scale - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_split_size() {
        let cfg = FlashDecodingConfig::new(1, 64, 100, 3).unwrap();
        assert_eq!(cfg.split_size(), 34); // ceil(100/3)
    }

    #[test]
    fn test_config_partial_grid_dim() {
        let cfg = FlashDecodingConfig::new(8, 64, 512, 4).unwrap();
        assert_eq!(cfg.partial_grid_dim(), (4, 8, 1));
    }

    #[test]
    fn test_config_merge_grid_dim() {
        let cfg = FlashDecodingConfig::new(8, 64, 512, 4).unwrap();
        assert_eq!(cfg.merge_grid_dim(), (8, 1, 1));
    }

    // ── GqaConfig tests ──────────────────────────────────────────────

    #[test]
    fn test_gqa_config_basic() {
        let gqa = GqaConfig::new(32, 8).unwrap();
        assert_eq!(gqa.group_size(), 4);
    }

    #[test]
    fn test_gqa_config_mha() {
        let gqa = GqaConfig::new(8, 8).unwrap();
        assert_eq!(gqa.group_size(), 1);
    }

    #[test]
    fn test_gqa_config_mqa() {
        let gqa = GqaConfig::new(8, 1).unwrap();
        assert_eq!(gqa.group_size(), 8);
    }

    #[test]
    fn test_gqa_config_rejects_zero_q() {
        assert!(GqaConfig::new(0, 8).is_err());
    }

    #[test]
    fn test_gqa_config_rejects_zero_kv() {
        assert!(GqaConfig::new(8, 0).is_err());
    }

    #[test]
    fn test_gqa_config_rejects_non_divisible() {
        assert!(GqaConfig::new(7, 3).is_err());
    }

    // ── split_kv_across_blocks tests ─────────────────────────────────

    #[test]
    fn test_split_kv_basic() {
        let ranges = split_kv_across_blocks(100, 4);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], (0, 25));
        assert_eq!(ranges[1], (25, 50));
        assert_eq!(ranges[2], (50, 75));
        assert_eq!(ranges[3], (75, 100));
    }

    #[test]
    fn test_split_kv_uneven() {
        let ranges = split_kv_across_blocks(10, 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], (0, 4));
        assert_eq!(ranges[1], (4, 8));
        assert_eq!(ranges[2], (8, 10));
    }

    #[test]
    fn test_split_kv_single() {
        let ranges = split_kv_across_blocks(50, 1);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (0, 50));
    }

    #[test]
    fn test_split_kv_more_splits_than_elements() {
        let ranges = split_kv_across_blocks(3, 10);
        // Each split gets 1 element, so only 3 non-empty ranges
        assert!(ranges.len() <= 10);
        let total: usize = ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_split_kv_zero_seq() {
        let ranges = split_kv_across_blocks(0, 4);
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_split_kv_zero_splits() {
        let ranges = split_kv_across_blocks(100, 0);
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_split_kv_seq_equals_one() {
        let ranges = split_kv_across_blocks(1, 4);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (0, 1));
    }

    #[test]
    fn test_split_kv_covers_full_range() {
        for seq in [1, 7, 64, 512, 2048, 8192] {
            for splits in [1, 2, 4, 8, 16] {
                let ranges = split_kv_across_blocks(seq, splits);
                assert_eq!(ranges.first().unwrap().0, 0);
                assert_eq!(ranges.last().unwrap().1, seq);
                // Contiguous: end of one == start of next
                for w in ranges.windows(2) {
                    assert_eq!(w[0].1, w[1].0);
                }
            }
        }
    }

    // ── partial_softmax tests ────────────────────────────────────────

    #[test]
    fn test_partial_softmax_single_element() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![3.0, 4.0];
        let (wv, _lse) = partial_softmax(&q, &k, &v, 0, 1, 2, 1.0);
        // Single element: weight = 1.0 after normalisation
        assert!((wv[0] - 3.0).abs() < 0.1);
        assert!((wv[1] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_partial_softmax_empty_range() {
        let q = vec![1.0, 0.0];
        let k: Vec<f32> = vec![];
        let v: Vec<f32> = vec![];
        let (wv, lse) = partial_softmax(&q, &k, &v, 0, 0, 2, 1.0);
        assert_eq!(wv, vec![0.0; 2]);
        assert!(lse.is_infinite() && lse.is_sign_negative());
    }

    #[test]
    fn test_partial_softmax_two_elements() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let (_wv, lse) = partial_softmax(&q, &k, &v, 0, 2, 2, 1.0);
        // lse should be finite
        assert!(lse.is_finite());
    }

    #[test]
    fn test_partial_softmax_lse_value() {
        // For a single element: lse ≈ score (since exp(0) = 1)
        let q = vec![1.0];
        let k = vec![2.0];
        let v = vec![5.0];
        let (_wv, lse) = partial_softmax(&q, &k, &v, 0, 1, 1, 1.0);
        // score = 2.0, lse = 2.0 + ln(1 + 1e-10) ≈ 2.0
        assert!((lse - 2.0).abs() < 0.01);
    }

    // ── merge_partial_attention tests ────────────────────────────────

    #[test]
    fn test_merge_single_partial() {
        let partial_out = vec![vec![1.0, 2.0, 3.0]];
        let partial_lse = vec![0.0];
        let merged = merge_partial_attention(&partial_out, &partial_lse, 3);
        // Single partial: output = partial / exp(0) = partial (after normalise)
        assert_close(&merged, &[1.0, 2.0, 3.0], 1e-4, "single partial merge");
    }

    #[test]
    fn test_merge_equal_lses() {
        // Two partials with the same LSE → average
        let p1 = vec![2.0, 4.0];
        let p2 = vec![6.0, 8.0];
        let lses = vec![1.0, 1.0];
        let merged = merge_partial_attention(&[p1, p2], &lses, 2);
        assert_close(&merged, &[4.0, 6.0], 1e-4, "equal LSE merge → average");
    }

    #[test]
    fn test_merge_empty() {
        let merged = merge_partial_attention(&[], &[], 4);
        assert_eq!(merged, vec![0.0; 4]);
    }

    #[test]
    fn test_merge_dominant_partial() {
        // One partial has much larger LSE → dominates
        let p1 = vec![1.0, 1.0];
        let p2 = vec![99.0, 99.0];
        let lses = vec![-100.0, 10.0];
        let merged = merge_partial_attention(&[p1, p2], &lses, 2);
        // p2 dominates
        assert_close(&merged, &[99.0, 99.0], 1e-2, "dominant partial merge");
    }

    // ── flash_decode_attention: basic correctness ────────────────────

    #[test]
    fn test_flash_decode_single_head_single_pos() {
        // seq_len=1, head_dim=2: trivially q @ k -> softmax([1.0]) -> v
        let q = vec![1.0, 0.5];
        let k = vec![0.3, 0.7];
        let v = vec![2.0, 3.0];
        let cfg = FlashDecodingConfig::new(1, 2, 1, 1).unwrap();
        let out = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        assert_close(&out, &v, 1e-4, "single head, seq=1");
    }

    #[test]
    fn test_flash_decode_matches_naive_small() {
        let mut seed = 42u32;
        let h = 1;
        let d = 4;
        let seq = 8;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "flash vs naive (1h, d=4, s=8)");
    }

    #[test]
    fn test_flash_decode_multi_head() {
        let mut seed = 123u32;
        let h = 4;
        let d = 8;
        let seq = 16;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "multi-head flash vs naive");
    }

    #[test]
    fn test_flash_decode_single_split() {
        let mut seed = 7u32;
        let h = 2;
        let d = 4;
        let seq = 32;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 1).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "single-split flash vs naive");
    }

    #[test]
    fn test_flash_decode_many_splits() {
        let mut seed = 55u32;
        let h = 2;
        let d = 4;
        let seq = 32;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 16).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "many-splits flash vs naive");
    }

    #[test]
    fn test_flash_decode_splits_equal_seq() {
        let mut seed = 99u32;
        let h = 1;
        let d = 4;
        let seq = 8;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        // num_splits == seq_len → each split has 1 element
        let cfg = FlashDecodingConfig::new(h, d, seq, seq).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-3, "splits == seq_len");
    }

    // ── Various sequence lengths ─────────────────────────────────────

    #[test]
    fn test_flash_decode_seq_1() {
        let mut seed = 10u32;
        let h = 2;
        let d = 4;
        let seq = 1;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 1).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "seq=1");
    }

    #[test]
    fn test_flash_decode_seq_64() {
        let mut seed = 20u32;
        let h = 4;
        let d = 32;
        let seq = 64;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "seq=64, d=32");
    }

    #[test]
    fn test_flash_decode_seq_512() {
        let mut seed = 30u32;
        let h = 2;
        let d = 64;
        let seq = 512;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 8).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "seq=512, d=64");
    }

    #[test]
    fn test_flash_decode_seq_2048() {
        let mut seed = 40u32;
        let h = 1;
        let d = 64;
        let seq = 2048;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 8).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-3, "seq=2048, d=64");
    }

    #[test]
    fn test_flash_decode_seq_8192() {
        let mut seed = 50u32;
        let h = 1;
        let d = 32;
        let seq = 8192;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 16).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-3, "seq=8192, d=32");
    }

    // ── Various head dimensions ──────────────────────────────────────

    #[test]
    fn test_flash_decode_head_dim_32() {
        let mut seed = 60u32;
        let h = 2;
        let d = 32;
        let seq = 64;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "d=32");
    }

    #[test]
    fn test_flash_decode_head_dim_64() {
        let mut seed = 61u32;
        let h = 2;
        let d = 64;
        let seq = 64;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "d=64");
    }

    #[test]
    fn test_flash_decode_head_dim_128() {
        let mut seed = 62u32;
        let h = 2;
        let d = 128;
        let seq = 64;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "d=128");
    }

    // ── Split & merge numerical equivalence ──────────────────────────

    #[test]
    fn test_split_merge_equivalence_to_naive() {
        // Manually split, compute partials, merge, compare to naive
        let mut seed = 70u32;
        let d = 8;
        let seq = 32;
        let scale = 1.0 / (d as f32).sqrt();

        let q = rand_vec(d, &mut seed);
        let k = rand_vec(seq * d, &mut seed);
        let v = rand_vec(seq * d, &mut seed);

        let splits = split_kv_across_blocks(seq, 4);
        let mut partial_outs = Vec::new();
        let mut partial_lses = Vec::new();

        for &(s, e) in &splits {
            let (wv, lse) = partial_softmax(&q, &k, &v, s, e, d, scale);
            partial_outs.push(wv);
            partial_lses.push(lse);
        }

        let merged = merge_partial_attention(&partial_outs, &partial_lses, d);
        let naive = naive_single_query_attention(&q, &k, &v, seq, d, scale);
        assert_close(&merged, &naive, 1e-4, "split+merge vs naive");
    }

    #[test]
    fn test_split_merge_two_partials() {
        let mut seed = 71u32;
        let d = 4;
        let seq = 16;
        let scale = 1.0 / (d as f32).sqrt();

        let q = rand_vec(d, &mut seed);
        let k = rand_vec(seq * d, &mut seed);
        let v = rand_vec(seq * d, &mut seed);

        let (p1, l1) = partial_softmax(&q, &k, &v, 0, 8, d, scale);
        let (p2, l2) = partial_softmax(&q, &k, &v, 8, 16, d, scale);
        let merged = merge_partial_attention(&[p1, p2], &[l1, l2], d);
        let naive = naive_single_query_attention(&q, &k, &v, seq, d, scale);
        assert_close(&merged, &naive, 1e-4, "two-partial merge vs naive");
    }

    #[test]
    fn test_split_merge_varying_splits() {
        let mut seed = 72u32;
        let d = 4;
        let seq = 64;
        let scale = 1.0 / (d as f32).sqrt();

        let q = rand_vec(d, &mut seed);
        let k = rand_vec(seq * d, &mut seed);
        let v = rand_vec(seq * d, &mut seed);

        let naive = naive_single_query_attention(&q, &k, &v, seq, d, scale);

        for num_splits in [1, 2, 4, 8, 16, 32, 64] {
            let splits = split_kv_across_blocks(seq, num_splits);
            let mut pos = Vec::new();
            let mut lses = Vec::new();
            for &(s, e) in &splits {
                let (wv, lse) = partial_softmax(&q, &k, &v, s, e, d, scale);
                pos.push(wv);
                lses.push(lse);
            }
            let merged = merge_partial_attention(&pos, &lses, d);
            assert_close(&merged, &naive, 1e-3, &format!("splits={num_splits}"));
        }
    }

    // ── Partial softmax LSE correctness ──────────────────────────────

    #[test]
    fn test_partial_lse_monotonic_with_higher_scores() {
        let q = vec![1.0];
        // k1 has score 1.0, k2 has score 2.0
        let k = vec![1.0, 2.0];
        let v = vec![0.0, 0.0];
        let (_, lse1) = partial_softmax(&q, &k, &v, 0, 1, 1, 1.0);
        let (_, lse2) = partial_softmax(&q, &k, &v, 1, 2, 1, 1.0);
        assert!(lse2 > lse1, "higher score → higher LSE");
    }

    #[test]
    fn test_partial_lse_additivity() {
        // log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.5, 0.5];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let (_, lse_full) = partial_softmax(&q, &k, &v, 0, 2, 2, 1.0);
        let (_, lse1) = partial_softmax(&q, &k, &v, 0, 1, 2, 1.0);
        let (_, lse2) = partial_softmax(&q, &k, &v, 1, 2, 2, 1.0);
        // log(exp(lse1) + exp(lse2)) should equal lse_full approximately
        let global_max = lse1.max(lse2);
        let combined = global_max + ((lse1 - global_max).exp() + (lse2 - global_max).exp()).ln();
        assert!((combined - lse_full).abs() < 0.05, "LSE additivity");
    }

    // ── ALiBi tests ──────────────────────────────────────────────────

    #[test]
    fn test_alibi_slope_values() {
        // h=0, n=8: slope = 2^(-8*1/8) = 2^(-1) = 0.5
        assert!((alibi_slope(0, 8) - 0.5).abs() < 1e-6);
        // h=7, n=8: slope = 2^(-8*8/8) = 2^(-8) = 1/256
        assert!((alibi_slope(7, 8) - 1.0 / 256.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_slopes_decrease() {
        let n = 16;
        let slopes: Vec<f32> = (0..n).map(|h| alibi_slope(h, n)).collect();
        for w in slopes.windows(2) {
            assert!(w[0] > w[1], "ALiBi slopes must decrease");
        }
    }

    #[test]
    fn test_flash_decode_alibi_output_shape() {
        let mut seed = 80u32;
        let h = 4;
        let d = 8;
        let seq = 16;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let out = flash_decode_with_alibi(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), h * d);
    }

    #[test]
    fn test_flash_decode_alibi_differs_from_standard() {
        let mut seed = 81u32;
        let h = 2;
        let d = 4;
        let seq = 16;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let alibi = flash_decode_with_alibi(&q, &k, &v, &cfg).unwrap();
        // They should differ (ALiBi adds position-dependent bias)
        let diff = max_abs_diff(&standard, &alibi);
        assert!(diff > 1e-6, "ALiBi should produce different output from standard");
    }

    #[test]
    fn test_flash_decode_alibi_recency_bias() {
        // With ALiBi, later positions get less negative bias → attended more.
        // Construct so that V values differ by position to verify.
        let h = 1;
        let d = 2;
        let seq = 8;

        let q = vec![1.0, 0.0];
        let k = vec![0.0_f32; seq * d]; // all-zero keys → equal base scores
        // Values: earlier = [1,0], later = [0,1]
        let mut v = vec![0.0_f32; seq * d];
        for j in 0..seq {
            if j < seq / 2 {
                v[j * d] = 1.0;
            } else {
                v[j * d + 1] = 1.0;
            }
        }

        let cfg = FlashDecodingConfig::new(h, d, seq, 1).unwrap();
        let out = flash_decode_with_alibi(&q, &k, &v, &cfg).unwrap();
        // ALiBi penalises distant (early) positions, so output should lean
        // towards later V values → out[1] > out[0].
        assert!(out[1] > out[0], "ALiBi should bias towards recent positions");
    }

    #[test]
    fn test_flash_decode_alibi_seq_1() {
        let q = vec![1.0, 0.5];
        let k = vec![0.3, 0.7];
        let v = vec![2.0, 3.0];
        let cfg = FlashDecodingConfig::new(1, 2, 1, 1).unwrap();
        let out = flash_decode_with_alibi(&q, &k, &v, &cfg).unwrap();
        // With seq=1, ALiBi bias at pos 0 is slope*(0-1+1) = 0, so same as standard
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        assert_close(&out, &standard, 1e-5, "ALiBi seq=1 matches standard");
    }

    // ── GQA tests ────────────────────────────────────────────────────

    #[test]
    fn test_gqa_mha_matches_standard() {
        // When num_q_heads == num_kv_heads, GQA == MHA == standard flash decode
        let mut seed = 90u32;
        let h = 4;
        let d = 8;
        let seq = 16;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let gqa = GqaConfig::new(h, h).unwrap();
        let gqa_out = flash_decode_gqa(&q, &k, &v, &cfg, &gqa).unwrap();
        let std_out = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        assert_close(&gqa_out, &std_out, 1e-5, "GQA with group=1 matches standard");
    }

    #[test]
    fn test_gqa_shared_kv_heads() {
        // 8 query heads, 2 KV heads → group size 4
        let mut seed = 91u32;
        let qh = 8;
        let kvh = 2;
        let d = 4;
        let seq = 16;
        let q = rand_vec(qh * d, &mut seed);
        let k = rand_vec(kvh * seq * d, &mut seed);
        let v = rand_vec(kvh * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(kvh, d, seq, 2).unwrap();
        let gqa = GqaConfig::new(qh, kvh).unwrap();
        let out = flash_decode_gqa(&q, &k, &v, &cfg, &gqa).unwrap();
        assert_eq!(out.len(), qh * d);

        // Q-heads 0..3 share KV-head 0, Q-heads 4..7 share KV-head 1
        // Within a group, different Q vectors produce different outputs
        let h0_out = &out[0..d];
        let h1_out = &out[d..2 * d];
        let diff = max_abs_diff(h0_out, h1_out);
        assert!(diff > 1e-6, "different q-heads in same group → different output");
    }

    #[test]
    fn test_gqa_mqa_single_kv_head() {
        // Multi-query attention: all q-heads share 1 KV head
        let mut seed = 92u32;
        let qh = 4;
        let kvh = 1;
        let d = 4;
        let seq = 8;
        let q = rand_vec(qh * d, &mut seed);
        let k = rand_vec(kvh * seq * d, &mut seed);
        let v = rand_vec(kvh * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(kvh, d, seq, 2).unwrap();
        let gqa = GqaConfig::new(qh, kvh).unwrap();
        let out = flash_decode_gqa(&q, &k, &v, &cfg, &gqa).unwrap();
        assert_eq!(out.len(), qh * d);
    }

    #[test]
    fn test_gqa_group_ratio_4() {
        let mut seed = 93u32;
        let qh = 32;
        let kvh = 8;
        let d = 32;
        let seq = 64;
        let q = rand_vec(qh * d, &mut seed);
        let k = rand_vec(kvh * seq * d, &mut seed);
        let v = rand_vec(kvh * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(kvh, d, seq, 4).unwrap();
        let gqa = GqaConfig::new(qh, kvh).unwrap();
        let out = flash_decode_gqa(&q, &k, &v, &cfg, &gqa).unwrap();
        assert_eq!(out.len(), qh * d);
    }

    #[test]
    fn test_gqa_rejects_bad_q_length() {
        let cfg = FlashDecodingConfig::new(2, 4, 8, 1).unwrap();
        let gqa = GqaConfig::new(4, 2).unwrap();
        let q = vec![0.0; 4]; // too short for 4 q-heads
        let k = vec![0.0; 2 * 8 * 4];
        let v = vec![0.0; 2 * 8 * 4];
        assert!(flash_decode_gqa(&q, &k, &v, &cfg, &gqa).is_err());
    }

    // ── Paged flash decode tests ─────────────────────────────────────

    fn make_paged_kv(
        k_contiguous: &[f32],
        v_contiguous: &[f32],
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        page_size: usize,
    ) -> (Vec<f32>, Vec<f32>, PagedKvDescriptor) {
        let num_pages = seq_len.div_ceil(page_size);
        let page_elems = page_size * head_dim;
        let per_head = num_pages * page_elems;
        let mut k_pages = vec![0.0_f32; num_heads * per_head];
        let mut v_pages = vec![0.0_f32; num_heads * per_head];

        // Identity page table: logical page i → physical page i
        let page_table: Vec<usize> = (0..num_pages).collect();

        for h in 0..num_heads {
            for lp in 0..num_pages {
                let tokens = page_size.min(seq_len - lp * page_size);
                for t in 0..tokens {
                    let src_pos = lp * page_size + t;
                    let src_off = h * seq_len * head_dim + src_pos * head_dim;
                    let dst_off = h * per_head + lp * page_elems + t * head_dim;
                    k_pages[dst_off..dst_off + head_dim]
                        .copy_from_slice(&k_contiguous[src_off..src_off + head_dim]);
                    v_pages[dst_off..dst_off + head_dim]
                        .copy_from_slice(&v_contiguous[src_off..src_off + head_dim]);
                }
            }
        }

        let desc = PagedKvDescriptor { page_size, page_table, num_physical_pages: num_pages };

        (k_pages, v_pages, desc)
    }

    #[test]
    fn test_paged_decode_matches_contiguous() {
        let mut seed = 100u32;
        let h = 2;
        let d = 4;
        let seq = 16;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();

        let (kp, vp, desc) = make_paged_kv(&k, &v, h, seq, d, 4);
        let paged = paged_flash_decode(&q, &kp, &vp, &cfg, &desc).unwrap();
        assert_close(&paged, &standard, 1e-4, "paged vs contiguous");
    }

    #[test]
    fn test_paged_decode_page_size_1() {
        let mut seed = 101u32;
        let h = 1;
        let d = 4;
        let seq = 8;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 1).unwrap();
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();

        let (kp, vp, desc) = make_paged_kv(&k, &v, h, seq, d, 1);
        let paged = paged_flash_decode(&q, &kp, &vp, &cfg, &desc).unwrap();
        assert_close(&paged, &standard, 1e-4, "paged page_size=1");
    }

    #[test]
    fn test_paged_decode_page_size_equals_seq() {
        let mut seed = 102u32;
        let h = 2;
        let d = 4;
        let seq = 16;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 1).unwrap();
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();

        let (kp, vp, desc) = make_paged_kv(&k, &v, h, seq, d, seq);
        let paged = paged_flash_decode(&q, &kp, &vp, &cfg, &desc).unwrap();
        assert_close(&paged, &standard, 1e-4, "paged page_size=seq");
    }

    #[test]
    fn test_paged_decode_page_size_16() {
        let mut seed = 103u32;
        let h = 2;
        let d = 8;
        let seq = 64;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();

        let (kp, vp, desc) = make_paged_kv(&k, &v, h, seq, d, 16);
        let paged = paged_flash_decode(&q, &kp, &vp, &cfg, &desc).unwrap();
        assert_close(&paged, &standard, 1e-4, "paged page_size=16");
    }

    #[test]
    fn test_paged_decode_page_size_256() {
        let mut seed = 104u32;
        let h = 1;
        let d = 4;
        let seq = 512;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let standard = flash_decode_attention(&q, &k, &v, &cfg).unwrap();

        let (kp, vp, desc) = make_paged_kv(&k, &v, h, seq, d, 256);
        let paged = paged_flash_decode(&q, &kp, &vp, &cfg, &desc).unwrap();
        assert_close(&paged, &standard, 1e-3, "paged page_size=256");
    }

    #[test]
    fn test_paged_decode_rejects_zero_page_size() {
        let q = vec![0.0; 4];
        let kp = vec![0.0; 64];
        let vp = vec![0.0; 64];
        let cfg = FlashDecodingConfig::new(1, 4, 8, 1).unwrap();
        let desc = PagedKvDescriptor { page_size: 0, page_table: vec![0], num_physical_pages: 1 };
        assert!(paged_flash_decode(&q, &kp, &vp, &cfg, &desc).is_err());
    }

    #[test]
    fn test_paged_decode_rejects_short_page_table() {
        let q = vec![0.0; 4];
        let kp = vec![0.0; 256];
        let vp = vec![0.0; 256];
        let cfg = FlashDecodingConfig::new(1, 4, 16, 1).unwrap();
        // need 4 pages (16/4) but only 1 in table
        let desc = PagedKvDescriptor { page_size: 4, page_table: vec![0], num_physical_pages: 4 };
        assert!(paged_flash_decode(&q, &kp, &vp, &cfg, &desc).is_err());
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_flash_decode_num_heads_1() {
        let mut seed = 110u32;
        let h = 1;
        let d = 4;
        let seq = 32;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let flash = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let naive = naive_multi_head(&q, &k, &v, h, d, seq, cfg.scale);
        assert_close(&flash, &naive, 1e-4, "num_heads=1");
    }

    #[test]
    fn test_flash_decode_rejects_short_q() {
        let cfg = FlashDecodingConfig::new(2, 4, 8, 1).unwrap();
        let q = vec![0.0; 4]; // need 2*4=8
        let k = vec![0.0; 2 * 8 * 4];
        let v = vec![0.0; 2 * 8 * 4];
        assert!(flash_decode_attention(&q, &k, &v, &cfg).is_err());
    }

    #[test]
    fn test_flash_decode_rejects_short_k() {
        let cfg = FlashDecodingConfig::new(1, 4, 8, 1).unwrap();
        let q = vec![0.0; 4];
        let k = vec![0.0; 16]; // need 32
        let v = vec![0.0; 32];
        assert!(flash_decode_attention(&q, &k, &v, &cfg).is_err());
    }

    #[test]
    fn test_flash_decode_rejects_short_v() {
        let cfg = FlashDecodingConfig::new(1, 4, 8, 1).unwrap();
        let q = vec![0.0; 4];
        let k = vec![0.0; 32];
        let v = vec![0.0; 16]; // need 32
        assert!(flash_decode_attention(&q, &k, &v, &cfg).is_err());
    }

    #[test]
    fn test_flash_decode_uniform_values() {
        // All V identical → output equals that V regardless of attention weights
        let h = 2;
        let d = 4;
        let seq = 16;
        let mut seed = 120u32;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v_val = [0.5, -0.3, 1.2, 0.0];
        let v: Vec<f32> = (0..h * seq).flat_map(|_| v_val.iter().copied()).collect();

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let out = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        for head in 0..h {
            assert_close(&out[head * d..(head + 1) * d], &v_val, 1e-4, "uniform V → output = V");
        }
    }

    #[test]
    fn test_flash_decode_deterministic() {
        let mut seed = 130u32;
        let h = 2;
        let d = 8;
        let seq = 64;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 4).unwrap();
        let out1 = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        let out2 = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out1, out2, "flash decode must be deterministic");
    }

    #[test]
    fn test_flash_decode_output_length() {
        let mut seed = 140u32;
        for (h, d) in [(1, 32), (4, 64), (8, 128)] {
            let seq = 64;
            let q = rand_vec(h * d, &mut seed);
            let k = rand_vec(h * seq * d, &mut seed);
            let v = rand_vec(h * seq * d, &mut seed);

            let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
            let out = flash_decode_attention(&q, &k, &v, &cfg).unwrap();
            assert_eq!(out.len(), h * d, "output length for h={h}, d={d}");
        }
    }

    // ── Cross-variant consistency ────────────────────────────────────

    #[test]
    fn test_alibi_rejects_short_inputs() {
        let cfg = FlashDecodingConfig::new(1, 4, 8, 1).unwrap();
        let q = vec![0.0; 2]; // too short
        let k = vec![0.0; 32];
        let v = vec![0.0; 32];
        assert!(flash_decode_with_alibi(&q, &k, &v, &cfg).is_err());
    }

    #[test]
    fn test_gqa_output_length() {
        let mut seed = 150u32;
        let qh = 16;
        let kvh = 4;
        let d = 8;
        let seq = 32;
        let q = rand_vec(qh * d, &mut seed);
        let k = rand_vec(kvh * seq * d, &mut seed);
        let v = rand_vec(kvh * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(kvh, d, seq, 2).unwrap();
        let gqa = GqaConfig::new(qh, kvh).unwrap();
        let out = flash_decode_gqa(&q, &k, &v, &cfg, &gqa).unwrap();
        assert_eq!(out.len(), qh * d);
    }

    #[test]
    fn test_paged_decode_output_length() {
        let mut seed = 160u32;
        let h = 4;
        let d = 8;
        let seq = 32;
        let q = rand_vec(h * d, &mut seed);
        let k = rand_vec(h * seq * d, &mut seed);
        let v = rand_vec(h * seq * d, &mut seed);

        let cfg = FlashDecodingConfig::new(h, d, seq, 2).unwrap();
        let (kp, vp, desc) = make_paged_kv(&k, &v, h, seq, d, 8);
        let out = paged_flash_decode(&q, &kp, &vp, &cfg, &desc).unwrap();
        assert_eq!(out.len(), h * d);
    }
}
