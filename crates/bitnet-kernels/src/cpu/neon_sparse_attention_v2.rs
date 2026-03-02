//! ARM NEON sparse attention v2 kernels for Apple Silicon.
//!
//! Provides six sparse attention patterns, each with a NEON-accelerated
//! fast path and a portable scalar fallback:
//!
//! 1. **Sliding window** — attend to the last `W` positions per query.
//! 2. **Block sparse** — attend within fixed-size blocks.
//! 3. **Local + global** — sliding window combined with global sentinel tokens.
//! 4. **Dilated** — attend to every N-th position.
//! 5. **Strided** — fixed stride pattern attention.
//! 6. **Top-k sparse** — attend only to the top-k scoring positions per query.
//!
//! All public functions accept flat row-major `&[f32]` tensors with shape
//! `[num_heads, seq_len, head_dim]` for Q/K/V and produce output of the
//! same shape.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const LANES: usize = 4;

// ── helpers ────────────────────────────────────────────────────────────

/// Scalar dot product of two slices.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// NEON-accelerated dot product.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0usize;
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    while i + LANES <= n {
        let va = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i)) };
        acc = unsafe { vfmaq_f32(acc, va, vb) };
        i += LANES;
    }
    let mut sum = unsafe { vaddvq_f32(acc) };
    for j in i..n {
        sum += a[j] * b[j];
    }
    sum
}

/// Choose the best available dot product.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: we are on aarch64 with NEON.
        unsafe { dot_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_scalar(a, b)
    }
}

/// Stable softmax (scalar) over `scores`, writing normalised probabilities
/// into `out`. Both slices must have the same length.
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum;
        }
    }
}

/// NEON-accelerated softmax in-place.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_inplace_neon(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let n = scores.len();
    // find max
    let mut i = 0usize;
    let mut vmax = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    while i + LANES <= n {
        let v = unsafe { vld1q_f32(scores.as_ptr().add(i)) };
        vmax = unsafe { vmaxq_f32(vmax, v) };
        i += LANES;
    }
    let mut max_val = unsafe { vmaxvq_f32(vmax) };
    for j in i..n {
        max_val = max_val.max(scores[j]);
    }
    // exp and sum
    i = 0;
    let vmax_bc = unsafe { vdupq_n_f32(max_val) };
    let mut vsum = unsafe { vdupq_n_f32(0.0) };
    while i + LANES <= n {
        let v = unsafe { vld1q_f32(scores.as_ptr().add(i)) };
        // scalar exp per lane (fast enough for moderate lengths)
        let mut tmp = [0.0f32; LANES];
        let diff = unsafe { vsubq_f32(v, vmax_bc) };
        unsafe { vst1q_f32(tmp.as_mut_ptr(), diff) };
        for t in &mut tmp {
            *t = t.exp();
        }
        let ve = unsafe { vld1q_f32(tmp.as_ptr()) };
        unsafe { vst1q_f32(scores.as_mut_ptr().add(i), ve) };
        vsum = unsafe { vaddq_f32(vsum, ve) };
        i += LANES;
    }
    let mut sum = unsafe { vaddvq_f32(vsum) };
    for j in i..n {
        let e = (scores[j] - max_val).exp();
        scores[j] = e;
        sum += e;
    }
    // normalise
    if sum > 0.0 {
        let inv = 1.0 / sum;
        let vinv = unsafe { vdupq_n_f32(inv) };
        i = 0;
        while i + LANES <= n {
            let v = unsafe { vld1q_f32(scores.as_ptr().add(i)) };
            let r = unsafe { vmulq_f32(v, vinv) };
            unsafe { vst1q_f32(scores.as_mut_ptr().add(i), r) };
            i += LANES;
        }
        for j in i..n {
            scores[j] *= inv;
        }
    }
}

/// Weighted sum: out[d] += weight * vec[d].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn weighted_add_neon(out: &mut [f32], vec: &[f32], weight: f32) {
    let n = out.len();
    debug_assert_eq!(n, vec.len());
    let vw = unsafe { vdupq_n_f32(weight) };
    let mut i = 0usize;
    while i + LANES <= n {
        let vo = unsafe { vld1q_f32(out.as_ptr().add(i)) };
        let vv = unsafe { vld1q_f32(vec.as_ptr().add(i)) };
        let r = unsafe { vfmaq_f32(vo, vv, vw) };
        unsafe { vst1q_f32(out.as_mut_ptr().add(i), r) };
        i += LANES;
    }
    for j in i..n {
        out[j] += weight * vec[j];
    }
}

fn weighted_add_scalar(out: &mut [f32], vec: &[f32], weight: f32) {
    for (o, v) in out.iter_mut().zip(vec.iter()) {
        *o += weight * *v;
    }
}

fn weighted_add(out: &mut [f32], vec: &[f32], weight: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { weighted_add_neon(out, vec, weight) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        weighted_add_scalar(out, vec, weight);
    }
}

// ── generic single-head attention with a position mask ─────────────────

/// Compute single-head attention for one head given a position-pair
/// predicate `mask_fn(query_pos, key_pos) -> bool` that returns `true`
/// when the pair should be attended.
///
/// * `q` — `[seq_len, head_dim]`
/// * `k` — `[seq_len, head_dim]`
/// * `v` — `[seq_len, head_dim]`
/// * `out` — `[seq_len, head_dim]` (written)
fn attend_masked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    mask_fn: &dyn Fn(usize, usize) -> bool,
) {
    for i in 0..seq_len {
        let qi = &q[i * head_dim..(i + 1) * head_dim];
        // Collect attended positions and their scores.
        let mut positions: Vec<usize> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        for j in 0..seq_len {
            if mask_fn(i, j) {
                let kj = &k[j * head_dim..(j + 1) * head_dim];
                let s = dot(qi, kj) * scale;
                positions.push(j);
                scores.push(s);
            }
        }
        let out_row = &mut out[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        if scores.is_empty() {
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { softmax_inplace_neon(&mut scores) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            softmax_inplace(&mut scores);
        }
        for (idx, &pos) in positions.iter().enumerate() {
            let vj = &v[pos * head_dim..(pos + 1) * head_dim];
            weighted_add(out_row, vj, scores[idx]);
        }
    }
}

/// Run `attend_masked` over all heads.
fn multi_head_attend_masked(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    mask_fn: impl Fn(usize, usize) -> bool,
) {
    let head_elems = seq_len * head_dim;
    for h in 0..num_heads {
        let offset = h * head_elems;
        attend_masked(
            &q[offset..offset + head_elems],
            &k[offset..offset + head_elems],
            &v[offset..offset + head_elems],
            &mut out[offset..offset + head_elems],
            seq_len,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            &mask_fn,
        );
    }
}

// ── 1. Sliding window attention ────────────────────────────────────────

/// Configuration for sliding window sparse attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowV2Config {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub window_size: usize,
    pub causal: bool,
}

/// NEON-accelerated sliding window attention.
///
/// Each query position `i` attends only to key positions `j` satisfying:
/// - causal: `j <= i && i - j < window_size`
/// - non-causal: `|i - j| < window_size`
pub fn sliding_window_attention_v2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &SlidingWindowV2Config,
) -> Vec<f32> {
    let SlidingWindowV2Config { num_heads, head_dim, seq_len, window_size, causal } = *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);
    let mut out = vec![0.0f32; total];
    let mask = move |i: usize, j: usize| -> bool {
        if causal {
            j <= i && (i - j) < window_size
        } else {
            let diff = if i >= j { i - j } else { j - i };
            diff < window_size
        }
    };
    multi_head_attend_masked(q, k, v, &mut out, num_heads, seq_len, head_dim, mask);
    out
}

/// Scalar fallback for sliding window attention (no NEON).
pub fn sliding_window_attention_v2_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &SlidingWindowV2Config,
) -> Vec<f32> {
    let SlidingWindowV2Config { num_heads, head_dim, seq_len, window_size, causal } = *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let mut positions = Vec::new();
            let mut scores = Vec::new();
            for j in 0..seq_len {
                let attend = if causal {
                    j <= i && (i - j) < window_size
                } else {
                    let d = if i >= j { i - j } else { j - i };
                    d < window_size
                };
                if attend {
                    let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                    scores.push(dot_scalar(qi, kj) * scale);
                    positions.push(j);
                }
            }
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            if scores.is_empty() {
                continue;
            }
            softmax_inplace(&mut scores);
            for (idx, &pos) in positions.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add_scalar(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

// ── 2. Block sparse attention ──────────────────────────────────────────

/// Configuration for block sparse attention.
#[derive(Debug, Clone)]
pub struct BlockSparseConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Block size (seq_len should ideally be a multiple of this).
    pub block_size: usize,
    pub causal: bool,
}

/// NEON-accelerated block sparse attention.
///
/// Positions within the same block attend to each other. Each position
/// `i` belongs to block `i / block_size`. When causal, position `i`
/// attends only to `j` in the same block where `j <= i`.
pub fn block_sparse_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &BlockSparseConfig,
) -> Vec<f32> {
    let BlockSparseConfig { num_heads, head_dim, seq_len, block_size, causal } = *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);
    let mut out = vec![0.0f32; total];
    let mask = move |i: usize, j: usize| -> bool {
        let same_block = i / block_size == j / block_size;
        if causal { same_block && j <= i } else { same_block }
    };
    multi_head_attend_masked(q, k, v, &mut out, num_heads, seq_len, head_dim, mask);
    out
}

/// Scalar fallback for block sparse attention.
pub fn block_sparse_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &BlockSparseConfig,
) -> Vec<f32> {
    let BlockSparseConfig { num_heads, head_dim, seq_len, block_size, causal } = *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let block_start = (i / block_size) * block_size;
            let block_end = (block_start + block_size).min(seq_len);
            let mut scores = Vec::new();
            let mut positions = Vec::new();
            for j in block_start..block_end {
                if causal && j > i {
                    continue;
                }
                let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                scores.push(dot_scalar(qi, kj) * scale);
                positions.push(j);
            }
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            if scores.is_empty() {
                continue;
            }
            softmax_inplace(&mut scores);
            for (idx, &pos) in positions.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add_scalar(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

// ── 3. Local + global attention ────────────────────────────────────────

/// Configuration for local + global attention.
#[derive(Debug, Clone)]
pub struct LocalGlobalConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Local window size.
    pub window_size: usize,
    /// Indices of global tokens (e.g. `[0]` for CLS).
    pub global_tokens: Vec<usize>,
    pub causal: bool,
}

/// NEON-accelerated local + global attention.
///
/// Each position attends to its local sliding window **plus** all global
/// sentinel tokens. Global tokens attend to every position.
pub fn local_global_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &LocalGlobalConfig,
) -> Vec<f32> {
    let LocalGlobalConfig { num_heads, head_dim, seq_len, window_size, ref global_tokens, causal } =
        *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);
    let mut out = vec![0.0f32; total];
    let globals = global_tokens.clone();
    let mask = move |i: usize, j: usize| -> bool {
        if causal && j > i {
            return false;
        }
        // global tokens attend everywhere; everyone attends to globals
        if globals.contains(&i) || globals.contains(&j) {
            return true;
        }
        // local window
        let diff = if i >= j { i - j } else { j - i };
        diff < window_size
    };
    multi_head_attend_masked(q, k, v, &mut out, num_heads, seq_len, head_dim, mask);
    out
}

/// Scalar fallback for local + global attention.
pub fn local_global_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &LocalGlobalConfig,
) -> Vec<f32> {
    let LocalGlobalConfig { num_heads, head_dim, seq_len, window_size, ref global_tokens, causal } =
        *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let mut scores = Vec::new();
            let mut positions = Vec::new();
            for j in 0..seq_len {
                if causal && j > i {
                    continue;
                }
                let is_global = global_tokens.contains(&i) || global_tokens.contains(&j);
                let in_window = {
                    let d = if i >= j { i - j } else { j - i };
                    d < window_size
                };
                if is_global || in_window {
                    let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                    scores.push(dot_scalar(qi, kj) * scale);
                    positions.push(j);
                }
            }
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            if scores.is_empty() {
                continue;
            }
            softmax_inplace(&mut scores);
            for (idx, &pos) in positions.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add_scalar(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

// ── 4. Dilated attention ───────────────────────────────────────────────

/// Configuration for dilated attention.
#[derive(Debug, Clone)]
pub struct DilatedConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Dilation rate: attend to every `dilation`-th position.
    pub dilation: usize,
    pub causal: bool,
}

/// NEON-accelerated dilated attention.
///
/// Position `i` attends to position `j` when `(i - j) % dilation == 0`
/// (optionally with causal constraint `j <= i`).
pub fn dilated_attention(q: &[f32], k: &[f32], v: &[f32], config: &DilatedConfig) -> Vec<f32> {
    let DilatedConfig { num_heads, head_dim, seq_len, dilation, causal } = *config;
    assert!(dilation > 0, "dilation must be > 0");
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);
    let mut out = vec![0.0f32; total];
    let mask = move |i: usize, j: usize| -> bool {
        if causal && j > i {
            return false;
        }
        let diff = if i >= j { i - j } else { j - i };
        diff % dilation == 0
    };
    multi_head_attend_masked(q, k, v, &mut out, num_heads, seq_len, head_dim, mask);
    out
}

/// Scalar fallback for dilated attention.
pub fn dilated_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &DilatedConfig,
) -> Vec<f32> {
    let DilatedConfig { num_heads, head_dim, seq_len, dilation, causal } = *config;
    assert!(dilation > 0);
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let mut scores = Vec::new();
            let mut positions = Vec::new();
            for j in 0..seq_len {
                if causal && j > i {
                    continue;
                }
                let diff = if i >= j { i - j } else { j - i };
                if diff % dilation == 0 {
                    let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                    scores.push(dot_scalar(qi, kj) * scale);
                    positions.push(j);
                }
            }
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            if scores.is_empty() {
                continue;
            }
            softmax_inplace(&mut scores);
            for (idx, &pos) in positions.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add_scalar(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

// ── 5. Strided attention ───────────────────────────────────────────────

/// Configuration for strided attention.
#[derive(Debug, Clone)]
pub struct StridedConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Stride: position `i` attends to positions `i`, `i ± stride`,
    /// `i ± 2·stride`, etc.
    pub stride: usize,
    pub causal: bool,
}

/// NEON-accelerated strided attention.
///
/// Position `i` attends to all positions `j` where `j % stride == i % stride`
/// (same stride class). Optionally causal.
pub fn strided_attention(q: &[f32], k: &[f32], v: &[f32], config: &StridedConfig) -> Vec<f32> {
    let StridedConfig { num_heads, head_dim, seq_len, stride, causal } = *config;
    assert!(stride > 0, "stride must be > 0");
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);
    let mut out = vec![0.0f32; total];
    let mask = move |i: usize, j: usize| -> bool {
        if causal && j > i {
            return false;
        }
        i % stride == j % stride
    };
    multi_head_attend_masked(q, k, v, &mut out, num_heads, seq_len, head_dim, mask);
    out
}

/// Scalar fallback for strided attention.
pub fn strided_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &StridedConfig,
) -> Vec<f32> {
    let StridedConfig { num_heads, head_dim, seq_len, stride, causal } = *config;
    assert!(stride > 0);
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let mut scores = Vec::new();
            let mut positions = Vec::new();
            for j in 0..seq_len {
                if causal && j > i {
                    continue;
                }
                if i % stride == j % stride {
                    let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                    scores.push(dot_scalar(qi, kj) * scale);
                    positions.push(j);
                }
            }
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            if scores.is_empty() {
                continue;
            }
            softmax_inplace(&mut scores);
            for (idx, &pos) in positions.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add_scalar(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

// ── 6. Top-k sparse attention ──────────────────────────────────────────

/// Configuration for top-k sparse attention.
#[derive(Debug, Clone)]
pub struct TopKSparseConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Number of highest-scoring key positions to attend to per query.
    pub top_k: usize,
    pub causal: bool,
}

/// NEON-accelerated top-k sparse attention.
///
/// For each query position, compute scores against all (valid) key
/// positions, then keep only the `top_k` highest and renormalise.
pub fn topk_sparse_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &TopKSparseConfig,
) -> Vec<f32> {
    let TopKSparseConfig { num_heads, head_dim, seq_len, top_k, causal } = *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    assert_eq!(k.len(), total);
    assert_eq!(v.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let mut scored: Vec<(usize, f32)> = Vec::new();
            for j in 0..seq_len {
                if causal && j > i {
                    continue;
                }
                let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                let s = dot(qi, kj) * scale;
                scored.push((j, s));
            }
            if scored.is_empty() {
                continue;
            }
            // partial sort: keep top_k
            let keep = top_k.min(scored.len());
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(keep);
            let mut scores: Vec<f32> = scored.iter().map(|&(_, s)| s).collect();
            #[cfg(target_arch = "aarch64")]
            {
                unsafe { softmax_inplace_neon(&mut scores) };
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                softmax_inplace(&mut scores);
            }
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            for (idx, &(pos, _)) in scored.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

/// Scalar fallback for top-k sparse attention.
pub fn topk_sparse_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &TopKSparseConfig,
) -> Vec<f32> {
    let TopKSparseConfig { num_heads, head_dim, seq_len, top_k, causal } = *config;
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let head_elems = seq_len * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let off = h * head_elems;
        for i in 0..seq_len {
            let qi = &q[off + i * head_dim..off + (i + 1) * head_dim];
            let mut scored: Vec<(usize, f32)> = Vec::new();
            for j in 0..seq_len {
                if causal && j > i {
                    continue;
                }
                let kj = &k[off + j * head_dim..off + (j + 1) * head_dim];
                let s = dot_scalar(qi, kj) * scale;
                scored.push((j, s));
            }
            if scored.is_empty() {
                continue;
            }
            let keep = top_k.min(scored.len());
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(keep);
            let mut scores: Vec<f32> = scored.iter().map(|&(_, s)| s).collect();
            softmax_inplace(&mut scores);
            let out_row = &mut out[off + i * head_dim..off + (i + 1) * head_dim];
            for (idx, &(pos, _)) in scored.iter().enumerate() {
                let vj = &v[off + pos * head_dim..off + (pos + 1) * head_dim];
                weighted_add_scalar(out_row, vj, scores[idx]);
            }
        }
    }
    out
}

// ── Dense reference (for testing) ──────────────────────────────────────

/// Full dense attention (causal or non-causal) used as the correctness
/// baseline in tests.
pub fn dense_attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let total = num_heads * seq_len * head_dim;
    assert_eq!(q.len(), total);
    let mut out = vec![0.0f32; total];
    let mask = move |i: usize, j: usize| -> bool { if causal { j <= i } else { true } };
    multi_head_attend_masked(q, k, v, &mut out, num_heads, seq_len, head_dim, mask);
    out
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    /// Simple deterministic pseudo-random data.
    fn make_data(n: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut s = seed;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.push(((s >> 33) as f32) / (u32::MAX as f32) - 0.5);
        }
        v
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        let diff = max_abs_diff(a, b);
        assert!(diff < tol, "{msg}: max_abs_diff = {diff} >= tol = {tol}");
    }

    // ── 1. Sliding window ──────────────────────────────────────────────

    #[test]
    fn sliding_window_neon_vs_scalar_seq16_w4() {
        let (nh, sl, hd, w) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 1);
        let k = make_data(nh * sl * hd, 2);
        let v = make_data(nh * sl * hd, 3);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw neon vs scalar seq16 w4");
    }

    #[test]
    fn sliding_window_neon_vs_scalar_seq32_w8() {
        let (nh, sl, hd, w) = (2, 32, 8, 8);
        let q = make_data(nh * sl * hd, 10);
        let k = make_data(nh * sl * hd, 11);
        let v = make_data(nh * sl * hd, 12);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw neon vs scalar seq32 w8");
    }

    #[test]
    fn sliding_window_neon_vs_scalar_seq64_w16() {
        let (nh, sl, hd, w) = (2, 64, 16, 16);
        let q = make_data(nh * sl * hd, 20);
        let k = make_data(nh * sl * hd, 21);
        let v = make_data(nh * sl * hd, 22);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: false,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw neon vs scalar seq64 w16 noncausal");
    }

    #[test]
    fn sliding_window_neon_vs_scalar_seq128_w32() {
        let (nh, sl, hd, w) = (1, 128, 8, 32);
        let q = make_data(nh * sl * hd, 30);
        let k = make_data(nh * sl * hd, 31);
        let v = make_data(nh * sl * hd, 32);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw seq128 w32");
    }

    #[test]
    fn sliding_window_full_window_equals_dense() {
        let (nh, sl, hd) = (1, 16, 8);
        let q = make_data(nh * sl * hd, 40);
        let k = make_data(nh * sl * hd, 41);
        let v = make_data(nh * sl * hd, 42);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: sl + 1,
            causal: true,
        };
        let sw = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        assert_close(&sw, &dense, 1e-4, "full window == dense causal");
    }

    #[test]
    fn sliding_window_seq1() {
        let (nh, sl, hd, w) = (1, 1, 8, 4);
        let q = make_data(nh * sl * hd, 50);
        let k = make_data(nh * sl * hd, 51);
        let v = make_data(nh * sl * hd, 52);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        // seq_len=1 → output equals v
        assert_close(&a, &v, 1e-5, "sw seq1");
    }

    #[test]
    fn sliding_window_window_leq_seqlen() {
        let (nh, sl, hd, w) = (1, 4, 8, 8);
        let q = make_data(nh * sl * hd, 55);
        let k = make_data(nh * sl * hd, 56);
        let v = make_data(nh * sl * hd, 57);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let sw = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        assert_close(&sw, &dense, 1e-4, "window >= seqlen => dense");
    }

    #[test]
    fn sliding_window_multi_head() {
        let (nh, sl, hd, w) = (4, 32, 8, 8);
        let q = make_data(nh * sl * hd, 60);
        let k = make_data(nh * sl * hd, 61);
        let v = make_data(nh * sl * hd, 62);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw multi-head");
    }

    #[test]
    fn sliding_window_noncausal_symmetric() {
        let (nh, sl, hd, w) = (1, 8, 4, 3);
        let q = make_data(nh * sl * hd, 70);
        let k = make_data(nh * sl * hd, 71);
        let v = make_data(nh * sl * hd, 72);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: false,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw noncausal");
    }

    #[test]
    fn sliding_window_seq256_w4() {
        let (nh, sl, hd, w) = (1, 256, 8, 4);
        let q = make_data(nh * sl * hd, 75);
        let k = make_data(nh * sl * hd, 76);
        let v = make_data(nh * sl * hd, 77);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw seq256 w4");
    }

    #[test]
    fn sliding_window_seq512_w16() {
        let (nh, sl, hd, w) = (1, 512, 8, 16);
        let q = make_data(nh * sl * hd, 80);
        let k = make_data(nh * sl * hd, 81);
        let v = make_data(nh * sl * hd, 82);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        let b = sliding_window_attention_v2_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "sw seq512 w16");
    }

    // ── 2. Block sparse ────────────────────────────────────────────────

    #[test]
    fn block_sparse_neon_vs_scalar_seq16_b4() {
        let (nh, sl, hd, bs) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 100);
        let k = make_data(nh * sl * hd, 101);
        let v = make_data(nh * sl * hd, 102);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: false,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse neon vs scalar");
    }

    #[test]
    fn block_sparse_neon_vs_scalar_seq32_b8() {
        let (nh, sl, hd, bs) = (2, 32, 8, 8);
        let q = make_data(nh * sl * hd, 110);
        let k = make_data(nh * sl * hd, 111);
        let v = make_data(nh * sl * hd, 112);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: true,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse causal seq32 b8");
    }

    #[test]
    fn block_sparse_seq64_b16() {
        let (nh, sl, hd, bs) = (1, 64, 16, 16);
        let q = make_data(nh * sl * hd, 120);
        let k = make_data(nh * sl * hd, 121);
        let v = make_data(nh * sl * hd, 122);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: false,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse seq64 b16");
    }

    #[test]
    fn block_sparse_block_eq_seqlen_is_dense() {
        let (nh, sl, hd) = (1, 16, 8);
        let q = make_data(nh * sl * hd, 130);
        let k = make_data(nh * sl * hd, 131);
        let v = make_data(nh * sl * hd, 132);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: sl,
            causal: false,
        };
        let bs = block_sparse_attention(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, false);
        assert_close(&bs, &dense, 1e-4, "block_size==seq_len => dense");
    }

    #[test]
    fn block_sparse_causal_seq128_b32() {
        let (nh, sl, hd, bs) = (1, 128, 8, 32);
        let q = make_data(nh * sl * hd, 135);
        let k = make_data(nh * sl * hd, 136);
        let v = make_data(nh * sl * hd, 137);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: true,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse causal seq128 b32");
    }

    #[test]
    fn block_sparse_seq1() {
        let (nh, sl, hd, bs) = (1, 1, 8, 4);
        let q = make_data(nh * sl * hd, 140);
        let k = make_data(nh * sl * hd, 141);
        let v = make_data(nh * sl * hd, 142);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: true,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        assert_close(&a, &v, 1e-5, "block sparse seq1");
    }

    #[test]
    fn block_sparse_multi_head() {
        let (nh, sl, hd, bs) = (4, 32, 8, 8);
        let q = make_data(nh * sl * hd, 145);
        let k = make_data(nh * sl * hd, 146);
        let v = make_data(nh * sl * hd, 147);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: true,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse multi-head");
    }

    #[test]
    fn block_sparse_seq256_b4() {
        let (nh, sl, hd, bs) = (1, 256, 8, 4);
        let q = make_data(nh * sl * hd, 150);
        let k = make_data(nh * sl * hd, 151);
        let v = make_data(nh * sl * hd, 152);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: false,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse seq256 b4");
    }

    // ── 3. Local + global ──────────────────────────────────────────────

    #[test]
    fn local_global_neon_vs_scalar_seq16_w4() {
        let (nh, sl, hd, w) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 200);
        let k = make_data(nh * sl * hd, 201);
        let v = make_data(nh * sl * hd, 202);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0],
            causal: true,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        let b = local_global_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "lg neon vs scalar");
    }

    #[test]
    fn local_global_neon_vs_scalar_seq32_w8() {
        let (nh, sl, hd, w) = (2, 32, 8, 8);
        let q = make_data(nh * sl * hd, 210);
        let k = make_data(nh * sl * hd, 211);
        let v = make_data(nh * sl * hd, 212);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0, sl - 1],
            causal: true,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        let b = local_global_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "lg seq32 w8");
    }

    #[test]
    fn local_global_no_globals_eq_sliding_window() {
        let (nh, sl, hd, w) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 220);
        let k = make_data(nh * sl * hd, 221);
        let v = make_data(nh * sl * hd, 222);
        let lg_cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![],
            causal: true,
        };
        let sw_cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let lg = local_global_attention(&q, &k, &v, &lg_cfg);
        let sw = sliding_window_attention_v2(&q, &k, &v, &sw_cfg);
        assert_close(&lg, &sw, 1e-5, "lg no globals == sliding window");
    }

    #[test]
    fn local_global_all_global_eq_dense() {
        let (nh, sl, hd, w) = (1, 8, 8, 2);
        let q = make_data(nh * sl * hd, 230);
        let k = make_data(nh * sl * hd, 231);
        let v = make_data(nh * sl * hd, 232);
        let all_global: Vec<usize> = (0..sl).collect();
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: all_global,
            causal: false,
        };
        let lg = local_global_attention(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, false);
        assert_close(&lg, &dense, 1e-4, "all global => dense");
    }

    #[test]
    fn local_global_seq1() {
        let (nh, sl, hd, w) = (1, 1, 8, 4);
        let q = make_data(nh * sl * hd, 235);
        let k = make_data(nh * sl * hd, 236);
        let v = make_data(nh * sl * hd, 237);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0],
            causal: true,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        assert_close(&a, &v, 1e-5, "lg seq1");
    }

    #[test]
    fn local_global_noncausal_seq64_w16() {
        let (nh, sl, hd, w) = (1, 64, 16, 16);
        let q = make_data(nh * sl * hd, 240);
        let k = make_data(nh * sl * hd, 241);
        let v = make_data(nh * sl * hd, 242);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0],
            causal: false,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        let b = local_global_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "lg noncausal seq64");
    }

    #[test]
    fn local_global_multi_head() {
        let (nh, sl, hd, w) = (4, 32, 8, 8);
        let q = make_data(nh * sl * hd, 250);
        let k = make_data(nh * sl * hd, 251);
        let v = make_data(nh * sl * hd, 252);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0],
            causal: true,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        let b = local_global_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "lg multi-head");
    }

    #[test]
    fn local_global_seq128_w32() {
        let (nh, sl, hd, w) = (1, 128, 8, 32);
        let q = make_data(nh * sl * hd, 255);
        let k = make_data(nh * sl * hd, 256);
        let v = make_data(nh * sl * hd, 257);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0, 1],
            causal: true,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        let b = local_global_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "lg seq128 w32");
    }

    // ── 4. Dilated ─────────────────────────────────────────────────────

    #[test]
    fn dilated_neon_vs_scalar_seq16_d2() {
        let (nh, sl, hd, d) = (1, 16, 8, 2);
        let q = make_data(nh * sl * hd, 300);
        let k = make_data(nh * sl * hd, 301);
        let v = make_data(nh * sl * hd, 302);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated neon vs scalar");
    }

    #[test]
    fn dilated_neon_vs_scalar_seq32_d4() {
        let (nh, sl, hd, d) = (2, 32, 8, 4);
        let q = make_data(nh * sl * hd, 310);
        let k = make_data(nh * sl * hd, 311);
        let v = make_data(nh * sl * hd, 312);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated seq32 d4");
    }

    #[test]
    fn dilated_dilation1_eq_dense() {
        let (nh, sl, hd) = (1, 16, 8);
        let q = make_data(nh * sl * hd, 320);
        let k = make_data(nh * sl * hd, 321);
        let v = make_data(nh * sl * hd, 322);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: 1, causal: true };
        let dil = dilated_attention(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        assert_close(&dil, &dense, 1e-4, "dilation=1 => dense");
    }

    #[test]
    fn dilated_seq1() {
        let (nh, sl, hd, d) = (1, 1, 8, 3);
        let q = make_data(nh * sl * hd, 330);
        let k = make_data(nh * sl * hd, 331);
        let v = make_data(nh * sl * hd, 332);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        assert_close(&a, &v, 1e-5, "dilated seq1");
    }

    #[test]
    fn dilated_noncausal_seq64_d4() {
        let (nh, sl, hd, d) = (1, 64, 16, 4);
        let q = make_data(nh * sl * hd, 340);
        let k = make_data(nh * sl * hd, 341);
        let v = make_data(nh * sl * hd, 342);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: false };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated noncausal seq64 d4");
    }

    #[test]
    fn dilated_multi_head() {
        let (nh, sl, hd, d) = (4, 32, 8, 2);
        let q = make_data(nh * sl * hd, 350);
        let k = make_data(nh * sl * hd, 351);
        let v = make_data(nh * sl * hd, 352);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated multi-head");
    }

    #[test]
    fn dilated_seq128_d8() {
        let (nh, sl, hd, d) = (1, 128, 8, 8);
        let q = make_data(nh * sl * hd, 355);
        let k = make_data(nh * sl * hd, 356);
        let v = make_data(nh * sl * hd, 357);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated seq128 d8");
    }

    #[test]
    fn dilated_seq256_d4() {
        let (nh, sl, hd, d) = (1, 256, 8, 4);
        let q = make_data(nh * sl * hd, 360);
        let k = make_data(nh * sl * hd, 361);
        let v = make_data(nh * sl * hd, 362);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated seq256 d4");
    }

    // ── 5. Strided ─────────────────────────────────────────────────────

    #[test]
    fn strided_neon_vs_scalar_seq16_s2() {
        let (nh, sl, hd, s) = (1, 16, 8, 2);
        let q = make_data(nh * sl * hd, 400);
        let k = make_data(nh * sl * hd, 401);
        let v = make_data(nh * sl * hd, 402);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided neon vs scalar");
    }

    #[test]
    fn strided_neon_vs_scalar_seq32_s4() {
        let (nh, sl, hd, s) = (2, 32, 8, 4);
        let q = make_data(nh * sl * hd, 410);
        let k = make_data(nh * sl * hd, 411);
        let v = make_data(nh * sl * hd, 412);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided seq32 s4");
    }

    #[test]
    fn strided_stride1_eq_dense() {
        let (nh, sl, hd) = (1, 16, 8);
        let q = make_data(nh * sl * hd, 420);
        let k = make_data(nh * sl * hd, 421);
        let v = make_data(nh * sl * hd, 422);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: 1, causal: true };
        let st = strided_attention(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        assert_close(&st, &dense, 1e-4, "stride=1 => dense");
    }

    #[test]
    fn strided_seq1() {
        let (nh, sl, hd, s) = (1, 1, 8, 3);
        let q = make_data(nh * sl * hd, 430);
        let k = make_data(nh * sl * hd, 431);
        let v = make_data(nh * sl * hd, 432);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        assert_close(&a, &v, 1e-5, "strided seq1");
    }

    #[test]
    fn strided_noncausal_seq64_s4() {
        let (nh, sl, hd, s) = (1, 64, 16, 4);
        let q = make_data(nh * sl * hd, 440);
        let k = make_data(nh * sl * hd, 441);
        let v = make_data(nh * sl * hd, 442);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: false };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided noncausal seq64");
    }

    #[test]
    fn strided_multi_head() {
        let (nh, sl, hd, s) = (4, 32, 8, 4);
        let q = make_data(nh * sl * hd, 450);
        let k = make_data(nh * sl * hd, 451);
        let v = make_data(nh * sl * hd, 452);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided multi-head");
    }

    #[test]
    fn strided_seq128_s8() {
        let (nh, sl, hd, s) = (1, 128, 8, 8);
        let q = make_data(nh * sl * hd, 455);
        let k = make_data(nh * sl * hd, 456);
        let v = make_data(nh * sl * hd, 457);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided seq128 s8");
    }

    #[test]
    fn strided_seq256_s4() {
        let (nh, sl, hd, s) = (1, 256, 8, 4);
        let q = make_data(nh * sl * hd, 460);
        let k = make_data(nh * sl * hd, 461);
        let v = make_data(nh * sl * hd, 462);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided seq256 s4");
    }

    // ── 6. Top-k sparse ────────────────────────────────────────────────

    #[test]
    fn topk_neon_vs_scalar_seq16_k4() {
        let (nh, sl, hd, tk) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 500);
        let k = make_data(nh * sl * hd, 501);
        let v = make_data(nh * sl * hd, 502);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk neon vs scalar");
    }

    #[test]
    fn topk_neon_vs_scalar_seq32_k8() {
        let (nh, sl, hd, tk) = (2, 32, 8, 8);
        let q = make_data(nh * sl * hd, 510);
        let k = make_data(nh * sl * hd, 511);
        let v = make_data(nh * sl * hd, 512);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk seq32 k8");
    }

    #[test]
    fn topk_k_ge_seqlen_eq_dense_causal() {
        let (nh, sl, hd) = (1, 16, 8);
        let q = make_data(nh * sl * hd, 520);
        let k = make_data(nh * sl * hd, 521);
        let v = make_data(nh * sl * hd, 522);
        let cfg = TopKSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            top_k: sl + 1,
            causal: true,
        };
        let tk = topk_sparse_attention(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        assert_close(&tk, &dense, 1e-4, "topk k>=seqlen => dense causal");
    }

    #[test]
    fn topk_k_ge_seqlen_eq_dense_noncausal() {
        let (nh, sl, hd) = (1, 16, 8);
        let q = make_data(nh * sl * hd, 525);
        let k = make_data(nh * sl * hd, 526);
        let v = make_data(nh * sl * hd, 527);
        let cfg = TopKSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            top_k: sl + 1,
            causal: false,
        };
        let tk = topk_sparse_attention(&q, &k, &v, &cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, false);
        assert_close(&tk, &dense, 1e-4, "topk k>=seqlen => dense noncausal");
    }

    #[test]
    fn topk_seq1() {
        let (nh, sl, hd, tk) = (1, 1, 8, 4);
        let q = make_data(nh * sl * hd, 530);
        let k = make_data(nh * sl * hd, 531);
        let v = make_data(nh * sl * hd, 532);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        assert_close(&a, &v, 1e-5, "topk seq1");
    }

    #[test]
    fn topk_noncausal_seq64_k16() {
        let (nh, sl, hd, tk) = (1, 64, 16, 16);
        let q = make_data(nh * sl * hd, 540);
        let k = make_data(nh * sl * hd, 541);
        let v = make_data(nh * sl * hd, 542);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: false };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk noncausal seq64 k16");
    }

    #[test]
    fn topk_multi_head() {
        let (nh, sl, hd, tk) = (4, 32, 8, 8);
        let q = make_data(nh * sl * hd, 550);
        let k = make_data(nh * sl * hd, 551);
        let v = make_data(nh * sl * hd, 552);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk multi-head");
    }

    #[test]
    fn topk_seq128_k4() {
        let (nh, sl, hd, tk) = (1, 128, 8, 4);
        let q = make_data(nh * sl * hd, 555);
        let k = make_data(nh * sl * hd, 556);
        let v = make_data(nh * sl * hd, 557);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk seq128 k4");
    }

    #[test]
    fn topk_seq256_k8() {
        let (nh, sl, hd, tk) = (1, 256, 8, 8);
        let q = make_data(nh * sl * hd, 560);
        let k = make_data(nh * sl * hd, 561);
        let v = make_data(nh * sl * hd, 562);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk seq256 k8");
    }

    // ── Cross-pattern tests ────────────────────────────────────────────

    #[test]
    fn dense_ref_causal_vs_noncausal_differ() {
        let (nh, sl, hd) = (1, 8, 4);
        let q = make_data(nh * sl * hd, 600);
        let k = make_data(nh * sl * hd, 601);
        let v = make_data(nh * sl * hd, 602);
        let c = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        let nc = dense_attention_reference(&q, &k, &v, nh, sl, hd, false);
        let diff = max_abs_diff(&c, &nc);
        assert!(diff > 1e-6, "causal and non-causal should differ");
    }

    #[test]
    fn sliding_window_stricter_than_dense() {
        let (nh, sl, hd, w) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 610);
        let k = make_data(nh * sl * hd, 611);
        let v = make_data(nh * sl * hd, 612);
        let sw_cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let sw = sliding_window_attention_v2(&q, &k, &v, &sw_cfg);
        let dense = dense_attention_reference(&q, &k, &v, nh, sl, hd, true);
        let diff = max_abs_diff(&sw, &dense);
        assert!(diff > 1e-6, "window < seq_len should differ from dense");
    }

    #[test]
    fn topk_k1_selects_single_value() {
        let (nh, sl, hd) = (1, 8, 4);
        let q = make_data(nh * sl * hd, 620);
        let k = make_data(nh * sl * hd, 621);
        let v = make_data(nh * sl * hd, 622);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: 1, causal: false };
        let tk = topk_sparse_attention(&q, &k, &v, &cfg);
        // Each row in the output should be a single value row from v
        // (softmax of a single score is 1.0).
        for i in 0..sl {
            let row = &tk[i * hd..(i + 1) * hd];
            let mut found = false;
            for j in 0..sl {
                let vr = &v[j * hd..(j + 1) * hd];
                if max_abs_diff(row, vr) < 1e-5 {
                    found = true;
                    break;
                }
            }
            assert!(found, "topk k=1 row {i} should match some v row");
        }
    }

    #[test]
    fn block_sparse_causal_first_pos_eq_v0() {
        let (nh, sl, hd, bs) = (1, 16, 8, 4);
        let q = make_data(nh * sl * hd, 630);
        let k = make_data(nh * sl * hd, 631);
        let v = make_data(nh * sl * hd, 632);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: true,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        // Position 0 with causal masking only attends to itself → output = v[0]
        assert_close(&a[..hd], &v[..hd], 1e-5, "causal pos0 == v[0]");
    }

    #[test]
    fn dilated_d_large_isolates_self() {
        let (nh, sl, hd) = (1, 8, 4);
        let q = make_data(nh * sl * hd, 640);
        let k = make_data(nh * sl * hd, 641);
        let v = make_data(nh * sl * hd, 642);
        // dilation >= seq_len means each position only attends to positions
        // with the same index mod dilation, which for dilation >= seq_len is just itself.
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: sl, causal: false };
        let a = dilated_attention(&q, &k, &v, &cfg);
        // Each position only sees itself → output = v
        assert_close(&a, &v, 1e-5, "dilation >= seqlen isolates to self");
    }

    #[test]
    fn strided_s_eq_seqlen_isolates_self() {
        let (nh, sl, hd) = (1, 8, 4);
        let q = make_data(nh * sl * hd, 650);
        let k = make_data(nh * sl * hd, 651);
        let v = make_data(nh * sl * hd, 652);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: sl, causal: false };
        let a = strided_attention(&q, &k, &v, &cfg);
        // stride == seq_len → each position in its own class → output = v
        assert_close(&a, &v, 1e-5, "stride == seqlen isolates to self");
    }

    #[test]
    fn sliding_window_causal_pos0_eq_v0() {
        let (nh, sl, hd, w) = (1, 32, 8, 4);
        let q = make_data(nh * sl * hd, 660);
        let k = make_data(nh * sl * hd, 661);
        let v = make_data(nh * sl * hd, 662);
        let cfg = SlidingWindowV2Config {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            causal: true,
        };
        let a = sliding_window_attention_v2(&q, &k, &v, &cfg);
        assert_close(&a[..hd], &v[..hd], 1e-5, "sw causal pos0 == v[0]");
    }

    #[test]
    fn topk_seq512_k4() {
        let (nh, sl, hd, tk) = (1, 512, 8, 4);
        let q = make_data(nh * sl * hd, 670);
        let k = make_data(nh * sl * hd, 671);
        let v = make_data(nh * sl * hd, 672);
        let cfg =
            TopKSparseConfig { num_heads: nh, head_dim: hd, seq_len: sl, top_k: tk, causal: true };
        let a = topk_sparse_attention(&q, &k, &v, &cfg);
        let b = topk_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "topk seq512 k4");
    }

    #[test]
    fn block_sparse_seq512_b8() {
        let (nh, sl, hd, bs) = (1, 512, 8, 8);
        let q = make_data(nh * sl * hd, 680);
        let k = make_data(nh * sl * hd, 681);
        let v = make_data(nh * sl * hd, 682);
        let cfg = BlockSparseConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            block_size: bs,
            causal: false,
        };
        let a = block_sparse_attention(&q, &k, &v, &cfg);
        let b = block_sparse_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "block sparse seq512 b8");
    }

    #[test]
    fn local_global_seq256_w8() {
        let (nh, sl, hd, w) = (1, 256, 8, 8);
        let q = make_data(nh * sl * hd, 690);
        let k = make_data(nh * sl * hd, 691);
        let v = make_data(nh * sl * hd, 692);
        let cfg = LocalGlobalConfig {
            num_heads: nh,
            head_dim: hd,
            seq_len: sl,
            window_size: w,
            global_tokens: vec![0],
            causal: true,
        };
        let a = local_global_attention(&q, &k, &v, &cfg);
        let b = local_global_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "lg seq256 w8");
    }

    #[test]
    fn strided_seq512_s8() {
        let (nh, sl, hd, s) = (1, 512, 8, 8);
        let q = make_data(nh * sl * hd, 700);
        let k = make_data(nh * sl * hd, 701);
        let v = make_data(nh * sl * hd, 702);
        let cfg =
            StridedConfig { num_heads: nh, head_dim: hd, seq_len: sl, stride: s, causal: true };
        let a = strided_attention(&q, &k, &v, &cfg);
        let b = strided_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "strided seq512 s8");
    }

    #[test]
    fn dilated_seq512_d8() {
        let (nh, sl, hd, d) = (1, 512, 8, 8);
        let q = make_data(nh * sl * hd, 710);
        let k = make_data(nh * sl * hd, 711);
        let v = make_data(nh * sl * hd, 712);
        let cfg =
            DilatedConfig { num_heads: nh, head_dim: hd, seq_len: sl, dilation: d, causal: true };
        let a = dilated_attention(&q, &k, &v, &cfg);
        let b = dilated_attention_scalar(&q, &k, &v, &cfg);
        assert_close(&a, &b, 1e-4, "dilated seq512 d8");
    }
}
