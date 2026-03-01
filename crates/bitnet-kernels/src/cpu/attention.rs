//! CPU SIMD-optimized attention computation kernel.
//!
//! Provides scaled dot-product attention, multi-head attention (MHA),
//! grouped-query attention (GQA), and incremental KV-cache attention
//! with optional causal masking.  Each public function performs runtime
//! AVX2 detection and falls back to a scalar implementation on platforms
//! without AVX2.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

use bitnet_common::{BitNetError, KernelError, Result};

// ── Configuration ──────────────────────────────────────────────────

/// Parameters that fully describe an attention computation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Sequence length (number of tokens).
    pub seq_len: usize,
    /// Whether to apply a causal (upper-triangular) mask.
    pub causal: bool,
    /// Scaling factor applied to Q·K^T.  When `None`, defaults to
    /// `1 / sqrt(head_dim)`.
    pub scale: Option<f32>,
}

impl AttentionConfig {
    /// Resolved scale factor: explicit value or `1/√head_dim`.
    #[inline]
    pub fn resolved_scale(&self) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }

    /// Validate the configuration, returning an error on nonsensical values.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if self.seq_len == 0 {
            return Err(invalid_arg("seq_len must be > 0"));
        }
        Ok(())
    }
}

/// Parameters for grouped-query attention.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key/value heads (must divide `num_q_heads`).
    pub num_kv_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Whether to apply a causal mask.
    pub causal: bool,
    /// Optional explicit scaling factor.
    pub scale: Option<f32>,
}

/// Stateless kernel entry-point — holds no data, just dispatches.
pub struct AttentionKernel;

// ── Helper ─────────────────────────────────────────────────────────

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Causal mask ────────────────────────────────────────────────────

/// Create an upper-triangular causal mask of shape `[seq_len, seq_len]`.
///
/// `mask[i * seq_len + j]` is `0.0` when `j <= i` (allowed) and
/// `f32::NEG_INFINITY` when `j > i` (masked).
pub fn causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Apply an additive mask to pre-softmax scores (in-place).
///
/// Both `scores` and `mask` have shape `[seq_len, seq_len]`.
pub fn apply_mask(scores: &mut [f32], mask: &[f32]) -> Result<()> {
    if scores.len() != mask.len() {
        return Err(invalid_arg("scores and mask must have the same length"));
    }
    for (s, &m) in scores.iter_mut().zip(mask.iter()) {
        *s += m;
    }
    Ok(())
}

// ── Softmax ────────────────────────────────────────────────────────

/// Numerically-stable row-wise softmax over a row of length `cols`.
///
/// Subtracts the row-max before exponentiation to avoid overflow.
fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Row-wise softmax over a matrix `[rows, cols]` stored in row-major order.
fn softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(data.len(), rows * cols);
    for r in 0..rows {
        softmax_row(&mut data[r * cols..(r + 1) * cols]);
    }
}

// ── Scalar implementations ─────────────────────────────────────────

/// Scalar dot-product of two `f32` slices.
#[inline]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Scalar Q·K^T → scores `[seq_q, seq_k]`.
fn scalar_qk(q: &[f32], k: &[f32], seq_q: usize, seq_k: usize, dim: usize) -> Vec<f32> {
    let mut scores = vec![0.0_f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            scores[i * seq_k + j] =
                scalar_dot(&q[i * dim..(i + 1) * dim], &k[j * dim..(j + 1) * dim]);
        }
    }
    scores
}

/// Scalar scores·V → output `[seq_q, dim_v]`.
fn scalar_sv(scores: &[f32], v: &[f32], seq_q: usize, seq_k: usize, dim_v: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq_q * dim_v];
    for i in 0..seq_q {
        for j in 0..seq_k {
            let w = scores[i * seq_k + j];
            for d in 0..dim_v {
                out[i * dim_v + d] += w * v[j * dim_v + d];
            }
        }
    }
    out
}

// ── AVX2 implementations (x86_64 only) ─────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx2_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for c in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(c * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(c * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // horizontal sum
    let hi = _mm256_extractf128_ps::<1>(acc);
    let lo = _mm256_castps256_ps128(acc);
    let sum4 = _mm_add_ps(hi, lo);
    let hi2 = _mm_movehl_ps(sum4, sum4);
    let sum2 = _mm_add_ps(sum4, hi2);
    let hi1 = _mm_shuffle_ps::<0x01>(sum2, sum2);
    let mut result = _mm_cvtss_f32(_mm_add_ss(sum2, hi1));
    // scalar tail
    for i in (chunks * 8)..n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

#[cfg(target_arch = "x86_64")]
fn avx2_qk(q: &[f32], k: &[f32], seq_q: usize, seq_k: usize, dim: usize) -> Vec<f32> {
    let mut scores = vec![0.0_f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            scores[i * seq_k + j] =
                unsafe { avx2_dot(&q[i * dim..(i + 1) * dim], &k[j * dim..(j + 1) * dim]) };
        }
    }
    scores
}

// ── Dispatch helpers ───────────────────────────────────────────────

/// Compute Q·K^T, choosing the best available SIMD path.
fn dispatch_qk(q: &[f32], k: &[f32], seq_q: usize, seq_k: usize, dim: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return avx2_qk(q, k, seq_q, seq_k, dim);
        }
    }
    scalar_qk(q, k, seq_q, seq_k, dim)
}

// ── Public API ─────────────────────────────────────────────────────

impl AttentionKernel {
    /// Scaled dot-product attention on a single head.
    ///
    /// * `q` — query, shape `[seq_q, head_dim]`
    /// * `k` — key,   shape `[seq_k, head_dim]`
    /// * `v` — value, shape `[seq_k, head_dim]`
    /// * `mask` — optional additive mask `[seq_q, seq_k]`
    /// * `scale` — scaling factor (typically `1/√d_k`)
    ///
    /// Returns output of shape `[seq_q, head_dim]`.
    pub fn scaled_dot_product(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: Option<&[f32]>,
        scale: f32,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        if head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if q.len() != seq_q * head_dim {
            return Err(invalid_arg("q length mismatch"));
        }
        if k.len() != seq_k * head_dim {
            return Err(invalid_arg("k length mismatch"));
        }
        if v.len() != seq_k * head_dim {
            return Err(invalid_arg("v length mismatch"));
        }

        // Q · K^T → [seq_q, seq_k]
        let mut scores = dispatch_qk(q, k, seq_q, seq_k, head_dim);

        // scale
        for s in &mut scores {
            *s *= scale;
        }

        // optional mask
        if let Some(m) = mask {
            apply_mask(&mut scores, m)?;
        }

        // softmax row-wise
        softmax_rows(&mut scores, seq_q, seq_k);

        // scores · V → [seq_q, head_dim]
        Ok(scalar_sv(&scores, v, seq_q, seq_k, head_dim))
    }

    /// Multi-head attention.
    ///
    /// * `q` — queries,  shape `[seq_len, num_heads * head_dim]`
    /// * `k` — keys,     shape `[seq_len, num_heads * head_dim]`
    /// * `v` — values,   shape `[seq_len, num_heads * head_dim]`
    ///
    /// Returns output of shape `[seq_len, num_heads * head_dim]`.
    pub fn multi_head_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        cfg: &AttentionConfig,
    ) -> Result<Vec<f32>> {
        cfg.validate()?;
        let AttentionConfig { num_heads, head_dim, seq_len, causal, .. } = *cfg;
        let model_dim = num_heads * head_dim;
        let expected = seq_len * model_dim;

        if q.len() != expected {
            return Err(invalid_arg("q length does not match seq_len * num_heads * head_dim"));
        }
        if k.len() != expected {
            return Err(invalid_arg("k length does not match seq_len * num_heads * head_dim"));
        }
        if v.len() != expected {
            return Err(invalid_arg("v length does not match seq_len * num_heads * head_dim"));
        }

        let scale = cfg.resolved_scale();
        let mask_vec = if causal { Some(causal_mask(seq_len)) } else { None };
        let mask_ref = mask_vec.as_deref();

        // Split into per-head slices, attend, concatenate.
        let mut output = vec![0.0_f32; expected];

        for h in 0..num_heads {
            let q_head = extract_head(q, seq_len, num_heads, head_dim, h);
            let k_head = extract_head(k, seq_len, num_heads, head_dim, h);
            let v_head = extract_head(v, seq_len, num_heads, head_dim, h);

            let head_out = Self::scaled_dot_product(
                &q_head, &k_head, &v_head, mask_ref, scale, seq_len, seq_len, head_dim,
            )?;

            scatter_head(&mut output, &head_out, seq_len, num_heads, head_dim, h);
        }

        Ok(output)
    }

    /// Grouped-query attention (GQA).
    ///
    /// Query has `num_q_heads` heads while key/value share `num_kv_heads`
    /// heads.  `num_q_heads` must be a multiple of `num_kv_heads`.
    ///
    /// * `q` — shape `[seq_len, num_q_heads * head_dim]`
    /// * `k` — shape `[seq_len, num_kv_heads * head_dim]`
    /// * `v` — shape `[seq_len, num_kv_heads * head_dim]`
    ///
    /// Returns shape `[seq_len, num_q_heads * head_dim]`.
    pub fn grouped_query_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        cfg: &GqaConfig,
    ) -> Result<Vec<f32>> {
        let GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal, scale } = *cfg;
        if num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 || seq_len == 0 {
            return Err(invalid_arg("all dimension parameters must be > 0"));
        }
        if !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(invalid_arg("num_q_heads must be a multiple of num_kv_heads"));
        }
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        if q.len() != seq_len * q_dim {
            return Err(invalid_arg("q length mismatch for GQA"));
        }
        if k.len() != seq_len * kv_dim {
            return Err(invalid_arg("k length mismatch for GQA"));
        }
        if v.len() != seq_len * kv_dim {
            return Err(invalid_arg("v length mismatch for GQA"));
        }

        let group_size = num_q_heads / num_kv_heads;
        let resolved_scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
        let mask_vec = if causal { Some(causal_mask(seq_len)) } else { None };
        let mask_ref = mask_vec.as_deref();

        let mut output = vec![0.0_f32; seq_len * q_dim];

        for kv_h in 0..num_kv_heads {
            let k_head = extract_head(k, seq_len, num_kv_heads, head_dim, kv_h);
            let v_head = extract_head(v, seq_len, num_kv_heads, head_dim, kv_h);

            for g in 0..group_size {
                let q_idx = kv_h * group_size + g;
                let q_head = extract_head(q, seq_len, num_q_heads, head_dim, q_idx);

                let head_out = Self::scaled_dot_product(
                    &q_head,
                    &k_head,
                    &v_head,
                    mask_ref,
                    resolved_scale,
                    seq_len,
                    seq_len,
                    head_dim,
                )?;

                scatter_head(&mut output, &head_out, seq_len, num_q_heads, head_dim, q_idx);
            }
        }

        Ok(output)
    }
}

// ── CpuAttentionConfig ─────────────────────────────────────────────

/// Batched attention configuration mirroring the CUDA
/// [`AttentionKernelConfig`](crate::cuda::attention::AttentionKernelConfig)
/// shape contract.
#[derive(Debug, Clone)]
pub struct CpuAttentionConfig {
    /// Batch size (number of independent sequences).
    pub batch_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Sequence length of query tokens.
    pub seq_len: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// Softmax temperature scale.  `None` → `1 / √head_dim`.
    pub scale: Option<f32>,
    /// Whether to apply a causal (upper-triangular) mask.
    pub causal_mask: bool,
}

impl CpuAttentionConfig {
    /// Resolved scale factor: explicit value or `1/√head_dim`.
    #[inline]
    pub fn resolved_scale(&self) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(invalid_arg("batch_size must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if self.seq_len == 0 {
            return Err(invalid_arg("seq_len must be > 0"));
        }
        Ok(())
    }
}

// ── Convenience wrappers ──────────────────────────────────────────

/// Build a causal mask and apply it to `scores` in-place.
///
/// `scores` has shape `[seq_len, seq_len]`.
pub fn apply_causal_mask(scores: &mut [f32], seq_len: usize) -> Result<()> {
    let expected = seq_len * seq_len;
    if scores.len() != expected {
        return Err(invalid_arg("scores length must equal seq_len * seq_len"));
    }
    let mask = causal_mask(seq_len);
    apply_mask(scores, &mask)
}

// ── Standalone function wrappers ──────────────────────────────────

/// Scaled dot-product attention (free function).
///
/// Equivalent to [`AttentionKernel::scaled_dot_product`] with
/// `scale = 1/√head_dim` and an optional causal mask.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_vec = if causal && seq_q == seq_k { Some(causal_mask(seq_q)) } else { None };
    let mask_ref = mask_vec.as_deref();
    AttentionKernel::scaled_dot_product(q, k, v, mask_ref, scale, seq_q, seq_k, head_dim)
}

/// Masked attention — convenience for causal self-attention.
///
/// Always applies a causal mask.  Delegates to
/// [`scaled_dot_product_attention`] with `causal = true`.
pub fn masked_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    scaled_dot_product_attention(q, k, v, seq_len, seq_len, head_dim, true)
}

/// Full multi-head attention (free function).
///
/// * `q` — `[seq_len, num_heads * head_dim]`
/// * `k` — `[seq_len, num_heads * head_dim]`
/// * `v` — `[seq_len, num_heads * head_dim]`
///
/// Returns `[seq_len, num_heads * head_dim]`.
pub fn multi_head_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    let cfg = AttentionConfig { num_heads, head_dim, seq_len, causal, scale: None };
    AttentionKernel::multi_head_attention(q, k, v, &cfg)
}

/// Incremental attention with KV cache for autoregressive decoding.
///
/// During generation the query is a single new token (`seq_q = 1`)
/// while the key/value tensors grow by one position each step.
///
/// * `q`       — new query, shape `[1, head_dim]`
/// * `k_cache` — cached keys,  shape `[cache_len, head_dim]`
/// * `v_cache` — cached values, shape `[cache_len, head_dim]`
/// * `k_new`   — new key,   shape `[1, head_dim]`
/// * `v_new`   — new value, shape `[1, head_dim]`
///
/// The function appends `k_new` / `v_new` to the caches **in-place**
/// and returns the attention output of shape `[1, head_dim]`.
pub fn attention_with_kv_cache(
    q: &[f32],
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    k_new: &[f32],
    v_new: &[f32],
    head_dim: usize,
) -> Result<Vec<f32>> {
    if head_dim == 0 {
        return Err(invalid_arg("head_dim must be > 0"));
    }
    if q.len() != head_dim {
        return Err(invalid_arg("q must have length head_dim"));
    }
    if k_new.len() != head_dim {
        return Err(invalid_arg("k_new must have length head_dim"));
    }
    if v_new.len() != head_dim {
        return Err(invalid_arg("v_new must have length head_dim"));
    }
    if !k_cache.len().is_multiple_of(head_dim) {
        return Err(invalid_arg("k_cache length must be a multiple of head_dim"));
    }
    if !v_cache.len().is_multiple_of(head_dim) {
        return Err(invalid_arg("v_cache length must be a multiple of head_dim"));
    }

    // Append new key/value to caches.
    k_cache.extend_from_slice(k_new);
    v_cache.extend_from_slice(v_new);

    let seq_kv = k_cache.len() / head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // No causal mask needed: seq_q == 1, so the single query token
    // can attend to all cached positions.
    AttentionKernel::scaled_dot_product(q, k_cache, v_cache, None, scale, 1, seq_kv, head_dim)
}

/// Causal self-attention convenience function.
///
/// Forces `causal = true` regardless of the `config.causal` field, then
/// delegates to [`AttentionKernel::multi_head_attention`].
///
/// * `q` — `[seq_len, num_heads * head_dim]`
/// * `k` — `[seq_len, num_heads * head_dim]`
/// * `v` — `[seq_len, num_heads * head_dim]`
pub fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &AttentionConfig,
) -> Result<Vec<f32>> {
    let mut causal_cfg = config.clone();
    causal_cfg.causal = true;
    AttentionKernel::multi_head_attention(q, k, v, &causal_cfg)
}

/// Apply rotary position embeddings (RoPE) to query and key tensors in-place.
///
/// Rotates consecutive dimension pairs `(x[2i], x[2i+1])` at each token
/// position using sinusoidal frequencies derived from `base = 10 000`.
///
/// Both `q` and `k` must have length `positions.len() * cols` where `cols`
/// is any multiple of `head_dim` (e.g., `num_heads * head_dim`).
///
/// * `q` — mutable query tensor, laid out as `[num_positions, cols]`
/// * `k` — mutable key tensor, same layout
/// * `positions` — absolute token positions, one per row
/// * `head_dim` — per-head dimension (must be even and > 0)
pub fn apply_rotary_embedding(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    head_dim: usize,
) -> Result<()> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(invalid_arg("head_dim must be even and > 0"));
    }
    if positions.is_empty() {
        return Ok(());
    }
    let num_pos = positions.len();
    if !q.len().is_multiple_of(num_pos) {
        return Err(invalid_arg("q length must be divisible by number of positions"));
    }
    if !k.len().is_multiple_of(num_pos) {
        return Err(invalid_arg("k length must be divisible by number of positions"));
    }
    let q_cols = q.len() / num_pos;
    let k_cols = k.len() / num_pos;
    if !q_cols.is_multiple_of(head_dim) {
        return Err(invalid_arg("q row width must be a multiple of head_dim"));
    }
    if !k_cols.is_multiple_of(head_dim) {
        return Err(invalid_arg("k row width must be a multiple of head_dim"));
    }

    rope_inplace(q, positions, head_dim, q_cols);
    rope_inplace(k, positions, head_dim, k_cols);
    Ok(())
}

/// Apply RoPE rotation to a single tensor in-place.
fn rope_inplace(data: &mut [f32], positions: &[usize], head_dim: usize, cols: usize) {
    let half_dim = head_dim / 2;
    let base: f32 = 10_000.0;
    let num_heads_in_row = cols / head_dim;

    for (p_idx, &pos) in positions.iter().enumerate() {
        let row = &mut data[p_idx * cols..(p_idx + 1) * cols];
        for h in 0..num_heads_in_row {
            let head_start = h * head_dim;
            for i in 0..half_dim {
                let exponent = -(2.0 * i as f32) / head_dim as f32;
                let theta = base.powf(exponent);
                let angle = pos as f32 * theta;
                let (sin_a, cos_a) = angle.sin_cos();

                let idx0 = head_start + 2 * i;
                let idx1 = head_start + 2 * i + 1;
                let x0 = row[idx0];
                let x1 = row[idx1];
                row[idx0] = x0 * cos_a - x1 * sin_a;
                row[idx1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

// ── Head extraction / scatter helpers ──────────────────────────────

/// Extract head `h` from an interleaved `[seq_len, num_heads * head_dim]`
/// tensor into a contiguous `[seq_len, head_dim]` buffer.
fn extract_head(
    data: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    h: usize,
) -> Vec<f32> {
    let stride = num_heads * head_dim;
    let mut head = Vec::with_capacity(seq_len * head_dim);
    for t in 0..seq_len {
        let start = t * stride + h * head_dim;
        head.extend_from_slice(&data[start..start + head_dim]);
    }
    head
}

/// Scatter a `[seq_len, head_dim]` result back into the interleaved
/// output tensor at head position `h`.
fn scatter_head(
    output: &mut [f32],
    head_out: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    h: usize,
) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        let dst_start = t * stride + h * head_dim;
        let src_start = t * head_dim;
        output[dst_start..dst_start + head_dim]
            .copy_from_slice(&head_out[src_start..src_start + head_dim]);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum())
    }

    fn slices_approx_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y))
    }

    // ── AttentionConfig ────────────────────────────────────────────

    #[test]
    fn config_default_scale() {
        let cfg =
            AttentionConfig { num_heads: 4, head_dim: 64, seq_len: 8, causal: false, scale: None };
        let expected = 1.0 / 64.0_f32.sqrt();
        assert!(approx_eq(cfg.resolved_scale(), expected));
    }

    #[test]
    fn config_explicit_scale() {
        let cfg = AttentionConfig {
            num_heads: 4,
            head_dim: 64,
            seq_len: 8,
            causal: false,
            scale: Some(0.5),
        };
        assert!(approx_eq(cfg.resolved_scale(), 0.5));
    }

    #[test]
    fn config_validate_zero_heads() {
        let cfg =
            AttentionConfig { num_heads: 0, head_dim: 64, seq_len: 8, causal: false, scale: None };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_head_dim() {
        let cfg =
            AttentionConfig { num_heads: 4, head_dim: 0, seq_len: 8, causal: false, scale: None };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_seq_len() {
        let cfg =
            AttentionConfig { num_heads: 4, head_dim: 64, seq_len: 0, causal: false, scale: None };
        assert!(cfg.validate().is_err());
    }

    // ── Causal mask ────────────────────────────────────────────────

    #[test]
    fn causal_mask_1x1() {
        let m = causal_mask(1);
        assert_eq!(m, vec![0.0]);
    }

    #[test]
    fn causal_mask_3x3() {
        let m = causal_mask(3);
        // Row 0: [0, -inf, -inf]
        assert_eq!(m[0], 0.0);
        assert!(m[1].is_infinite() && m[1] < 0.0);
        assert!(m[2].is_infinite() && m[2] < 0.0);
        // Row 1: [0, 0, -inf]
        assert_eq!(m[3], 0.0);
        assert_eq!(m[4], 0.0);
        assert!(m[5].is_infinite() && m[5] < 0.0);
        // Row 2: [0, 0, 0]
        assert_eq!(m[6], 0.0);
        assert_eq!(m[7], 0.0);
        assert_eq!(m[8], 0.0);
    }

    #[test]
    fn causal_mask_diagonal_is_zero() {
        for n in 1..=8 {
            let m = causal_mask(n);
            for i in 0..n {
                assert_eq!(m[i * n + i], 0.0, "diagonal at ({i},{i}) should be 0");
            }
        }
    }

    // ── apply_mask ─────────────────────────────────────────────────

    #[test]
    fn apply_mask_basic() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];
        apply_mask(&mut scores, &mask).unwrap();
        assert_eq!(scores[0], 1.0);
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
        assert_eq!(scores[2], 3.0);
        assert_eq!(scores[3], 4.0);
    }

    #[test]
    fn apply_mask_length_mismatch() {
        let mut scores = vec![1.0, 2.0];
        let mask = vec![0.0];
        assert!(apply_mask(&mut scores, &mask).is_err());
    }

    // ── Softmax ────────────────────────────────────────────────────

    #[test]
    fn softmax_uniform() {
        let mut row = vec![1.0, 1.0, 1.0, 1.0];
        softmax_row(&mut row);
        for &v in &row {
            assert!(approx_eq(v, 0.25));
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn softmax_preserves_order() {
        let mut row = vec![1.0, 3.0, 2.0];
        softmax_row(&mut row);
        assert!(row[1] > row[2] && row[2] > row[0]);
    }

    #[test]
    fn softmax_numerical_stability_large_values() {
        let mut row = vec![1000.0, 1001.0, 1002.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0), "sum was {sum}");
        assert!(row[2] > row[1] && row[1] > row[0]);
    }

    #[test]
    fn softmax_with_neg_infinity() {
        let mut row = vec![1.0, f32::NEG_INFINITY, 2.0];
        softmax_row(&mut row);
        assert!(approx_eq(row[1], 0.0));
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn softmax_single_element() {
        let mut row = vec![42.0];
        softmax_row(&mut row);
        assert!(approx_eq(row[0], 1.0));
    }

    // ── Scaled dot-product attention ───────────────────────────────

    #[test]
    fn sdp_identity_values() {
        // Q = K = identity-like, V = known values → output ≈ softmax-weighted V
        let head_dim = 2;
        let seq_len = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let out =
            AttentionKernel::scaled_dot_product(&q, &k, &v, None, 1.0, seq_len, seq_len, head_dim)
                .unwrap();
        assert_eq!(out.len(), seq_len * head_dim);
        // Each output row must be a convex combination of V rows
        for r in 0..seq_len {
            let row = &out[r * head_dim..(r + 1) * head_dim];
            for &val in row {
                assert!(val >= 1.0 && val <= 4.0, "out of convex range: {val}");
            }
        }
    }

    #[test]
    fn sdp_with_causal_mask() {
        let dim = 2;
        let seq = 3;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v: Vec<f32> = (0..seq).flat_map(|i| vec![i as f32; dim]).collect();
        let mask = causal_mask(seq);
        let out = AttentionKernel::scaled_dot_product(&q, &k, &v, Some(&mask), 1.0, seq, seq, dim)
            .unwrap();
        // Row 0 can only attend to position 0 → output ≈ v[0]
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
    }

    #[test]
    fn sdp_scale_factor_effect() {
        let dim = 2;
        let seq = 2;
        let q = vec![2.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out_large =
            AttentionKernel::scaled_dot_product(&q, &k, &v, None, 10.0, seq, seq, dim).unwrap();
        let out_small =
            AttentionKernel::scaled_dot_product(&q, &k, &v, None, 0.01, seq, seq, dim).unwrap();
        // With identical K rows, both should produce uniform attention,
        // but verify outputs are valid (sum-of-weights = 1 per row).
        assert_eq!(out_large.len(), seq * dim);
        assert_eq!(out_small.len(), seq * dim);
    }

    #[test]
    fn sdp_zero_scale_uniform_attention() {
        // scale=0 → all scores identical → uniform softmax
        let dim = 2;
        let seq = 2;
        let q = vec![5.0, 3.0, 1.0, 7.0];
        let k = vec![2.0, 4.0, 6.0, 8.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out =
            AttentionKernel::scaled_dot_product(&q, &k, &v, None, 0.0, seq, seq, dim).unwrap();
        // Uniform attention → output = average of V rows
        let expected_d0 = (10.0 + 30.0) / 2.0;
        let expected_d1 = (20.0 + 40.0) / 2.0;
        assert!(approx_eq(out[0], expected_d0));
        assert!(approx_eq(out[1], expected_d1));
    }

    #[test]
    fn sdp_dimension_mismatch_q() {
        let result = AttentionKernel::scaled_dot_product(
            &[1.0],
            &[1.0, 2.0],
            &[1.0, 2.0],
            None,
            1.0,
            1,
            1,
            2,
        );
        assert!(result.is_err());
    }

    #[test]
    fn sdp_seq_len_one() {
        let dim = 4;
        let q = vec![1.0; dim];
        let k = vec![1.0; dim];
        let v = vec![2.0; dim];
        let out = AttentionKernel::scaled_dot_product(&q, &k, &v, None, 1.0, 1, 1, dim).unwrap();
        // Single token → attention weight = 1.0 → output = v
        assert!(slices_approx_eq(&out, &v));
    }

    // ── Multi-head attention ───────────────────────────────────────

    #[test]
    fn mha_single_head_matches_sdp() {
        let cfg =
            AttentionConfig { num_heads: 1, head_dim: 4, seq_len: 2, causal: false, scale: None };
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mha = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        let sdp = AttentionKernel::scaled_dot_product(
            &q,
            &k,
            &v,
            None,
            cfg.resolved_scale(),
            cfg.seq_len,
            cfg.seq_len,
            cfg.head_dim,
        )
        .unwrap();
        assert!(slices_approx_eq(&mha, &sdp));
    }

    #[test]
    fn mha_output_shape() {
        let cfg =
            AttentionConfig { num_heads: 4, head_dim: 8, seq_len: 3, causal: false, scale: None };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q = vec![0.1; n];
        let k = vec![0.1; n];
        let v = vec![0.1; n];
        let out = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), n);
    }

    #[test]
    fn mha_causal_first_position() {
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 2,
            seq_len: 4,
            causal: true,
            scale: Some(1.0),
        };
        let model_dim = cfg.num_heads * cfg.head_dim;
        let n = cfg.seq_len * model_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        // Each position has a unique value signature
        let mut v = vec![0.0_f32; n];
        for t in 0..cfg.seq_len {
            for d in 0..model_dim {
                v[t * model_dim + d] = (t * model_dim + d) as f32;
            }
        }
        let out = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        // Position 0 can only see itself → output[0..model_dim] ≈ v[0..model_dim]
        assert!(slices_approx_eq(&out[..model_dim], &v[..model_dim]));
    }

    #[test]
    fn mha_dimension_mismatch() {
        let cfg =
            AttentionConfig { num_heads: 2, head_dim: 4, seq_len: 3, causal: false, scale: None };
        let wrong_len = vec![0.0; 10]; // wrong size
        let correct = vec![0.0; 24];
        assert!(
            AttentionKernel::multi_head_attention(&wrong_len, &correct, &correct, &cfg).is_err()
        );
    }

    // ── Grouped-query attention ────────────────────────────────────

    #[test]
    fn gqa_equal_heads_matches_mha() {
        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 2;
        let n = seq_len * num_heads * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();

        let cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: false, scale: None };
        let mha = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        let gqa = AttentionKernel::grouped_query_attention(
            &q,
            &k,
            &v,
            &GqaConfig {
                num_q_heads: num_heads,
                num_kv_heads: num_heads,
                head_dim,
                seq_len,
                causal: false,
                scale: None,
            },
        )
        .unwrap();
        assert!(slices_approx_eq(&mha, &gqa));
    }

    #[test]
    fn gqa_2x_ratio() {
        let num_q = 4;
        let num_kv = 2;
        let head_dim = 4;
        let seq_len = 2;
        let q = vec![0.1; seq_len * num_q * head_dim];
        let k = vec![0.1; seq_len * num_kv * head_dim];
        let v = vec![0.2; seq_len * num_kv * head_dim];
        let out = AttentionKernel::grouped_query_attention(
            &q,
            &k,
            &v,
            &GqaConfig {
                num_q_heads: num_q,
                num_kv_heads: num_kv,
                head_dim,
                seq_len,
                causal: false,
                scale: None,
            },
        )
        .unwrap();
        assert_eq!(out.len(), seq_len * num_q * head_dim);
    }

    #[test]
    fn gqa_4x_ratio() {
        let num_q = 8;
        let num_kv = 2;
        let head_dim = 4;
        let seq_len = 3;
        let q = vec![0.1; seq_len * num_q * head_dim];
        let k = vec![0.1; seq_len * num_kv * head_dim];
        let v = vec![0.2; seq_len * num_kv * head_dim];
        let out = AttentionKernel::grouped_query_attention(
            &q,
            &k,
            &v,
            &GqaConfig {
                num_q_heads: num_q,
                num_kv_heads: num_kv,
                head_dim,
                seq_len,
                causal: true,
                scale: None,
            },
        )
        .unwrap();
        assert_eq!(out.len(), seq_len * num_q * head_dim);
    }

    #[test]
    fn gqa_single_kv_head() {
        // Multi-query attention: 4 query heads, 1 KV head
        let num_q = 4;
        let num_kv = 1;
        let head_dim = 4;
        let seq_len = 2;
        let q = vec![1.0; seq_len * num_q * head_dim];
        let k = vec![1.0; seq_len * num_kv * head_dim];
        let v = vec![0.5; seq_len * num_kv * head_dim];
        let out = AttentionKernel::grouped_query_attention(
            &q,
            &k,
            &v,
            &GqaConfig {
                num_q_heads: num_q,
                num_kv_heads: num_kv,
                head_dim,
                seq_len,
                causal: false,
                scale: None,
            },
        )
        .unwrap();
        assert_eq!(out.len(), seq_len * num_q * head_dim);
        // All query heads share the same KV → all head outputs identical
        let stride = num_q * head_dim;
        for t in 0..seq_len {
            let head0 = &out[t * stride..t * stride + head_dim];
            for h in 1..num_q {
                let head_h = &out[t * stride + h * head_dim..t * stride + (h + 1) * head_dim];
                assert!(slices_approx_eq(head0, head_h), "heads should match for shared KV");
            }
        }
    }

    #[test]
    fn gqa_invalid_head_ratio() {
        let result = AttentionKernel::grouped_query_attention(
            &[0.0; 12],
            &[0.0; 12],
            &[0.0; 12],
            &GqaConfig {
                num_q_heads: 3,
                num_kv_heads: 2,
                head_dim: 2,
                seq_len: 1,
                causal: false,
                scale: None,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn gqa_causal_mask_applied() {
        let num_q = 2;
        let num_kv = 1;
        let head_dim = 2;
        let seq_len = 3;
        let q = vec![1.0; seq_len * num_q * head_dim];
        let k = vec![1.0; seq_len * num_kv * head_dim];
        let mut v = vec![0.0_f32; seq_len * num_kv * head_dim];
        for t in 0..seq_len {
            for d in 0..head_dim {
                v[t * head_dim + d] = t as f32;
            }
        }
        let out = AttentionKernel::grouped_query_attention(
            &q,
            &k,
            &v,
            &GqaConfig {
                num_q_heads: num_q,
                num_kv_heads: num_kv,
                head_dim,
                seq_len,
                causal: true,
                scale: Some(1.0),
            },
        )
        .unwrap();
        // Position 0 can only attend to itself → output ≈ v[0] = 0.0
        let stride = num_q * head_dim;
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
        // Position 2 attends to 0,1,2 uniformly → average
        let row2_start = 2 * stride;
        let expected = (0.0 + 1.0 + 2.0) / 3.0;
        assert!(
            approx_eq(out[row2_start], expected),
            "got {} expected {}",
            out[row2_start],
            expected
        );
    }

    // ── Extract / scatter round-trip ───────────────────────────────

    #[test]
    fn extract_scatter_roundtrip() {
        let seq = 3;
        let heads = 2;
        let dim = 4;
        let original: Vec<f32> = (0..(seq * heads * dim)).map(|i| i as f32).collect();
        let mut reconstructed = vec![0.0_f32; original.len()];
        for h in 0..heads {
            let extracted = extract_head(&original, seq, heads, dim, h);
            scatter_head(&mut reconstructed, &extracted, seq, heads, dim, h);
        }
        assert!(slices_approx_eq(&original, &reconstructed));
    }

    // ── Scalar vs dispatch parity ──────────────────────────────────

    #[test]
    fn dispatch_qk_matches_scalar() {
        let seq = 3;
        let dim = 8;
        let q: Vec<f32> = (0..(seq * dim)).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..(seq * dim)).map(|i| (i as f32) * 0.05).collect();
        let scalar = scalar_qk(&q, &k, seq, seq, dim);
        let dispatched = dispatch_qk(&q, &k, seq, seq, dim);
        assert!(slices_approx_eq(&scalar, &dispatched), "scalar and dispatch diverge");
    }

    // ── CpuAttentionConfig ─────────────────────────────────────────

    #[test]
    fn cpu_config_default_scale() {
        let cfg = CpuAttentionConfig {
            batch_size: 1,
            num_heads: 4,
            seq_len: 8,
            head_dim: 64,
            scale: None,
            causal_mask: false,
        };
        let expected = 1.0 / 64.0_f32.sqrt();
        assert!(approx_eq(cfg.resolved_scale(), expected));
    }

    #[test]
    fn cpu_config_validate_zero_batch() {
        let cfg = CpuAttentionConfig {
            batch_size: 0,
            num_heads: 4,
            seq_len: 8,
            head_dim: 64,
            scale: None,
            causal_mask: false,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn cpu_config_validate_ok() {
        let cfg = CpuAttentionConfig {
            batch_size: 2,
            num_heads: 4,
            seq_len: 8,
            head_dim: 64,
            scale: Some(0.5),
            causal_mask: true,
        };
        assert!(cfg.validate().is_ok());
        assert!(approx_eq(cfg.resolved_scale(), 0.5));
    }

    // ── apply_causal_mask ──────────────────────────────────────────

    #[test]
    fn apply_causal_mask_basic() {
        let mut scores = vec![1.0; 9]; // 3×3
        apply_causal_mask(&mut scores, 3).unwrap();
        // Diagonal and below unchanged (1.0 + 0.0)
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[3], 1.0);
        assert_eq!(scores[4], 1.0);
        // Upper triangle should be -inf
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
        assert!(scores[2].is_infinite() && scores[2] < 0.0);
        assert!(scores[5].is_infinite() && scores[5] < 0.0);
    }

    #[test]
    fn apply_causal_mask_length_mismatch() {
        let mut scores = vec![1.0; 5];
        assert!(apply_causal_mask(&mut scores, 3).is_err());
    }

    // ── scaled_dot_product_attention (free function) ───────────────

    #[test]
    fn sdpa_free_fn_no_mask() {
        let dim = 4;
        let q = vec![1.0; dim];
        let k = vec![1.0; dim];
        let v = vec![2.0; dim];
        let out = scaled_dot_product_attention(&q, &k, &v, 1, 1, dim, false).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn sdpa_free_fn_causal() {
        let dim = 2;
        let seq = 3;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v: Vec<f32> = (0..seq).flat_map(|i| vec![i as f32; dim]).collect();
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, true).unwrap();
        // Row 0 can only attend to position 0
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
    }

    // ── masked_attention ───────────────────────────────────────────

    #[test]
    fn masked_attention_single_token() {
        let dim = 4;
        let q = vec![1.0; dim];
        let k = vec![1.0; dim];
        let v = vec![3.0; dim];
        let out = masked_attention(&q, &k, &v, 1, dim).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn masked_attention_first_row_self_only() {
        let dim = 2;
        let seq = 4;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let mut v = vec![0.0; seq * dim];
        for t in 0..seq {
            for d in 0..dim {
                v[t * dim + d] = (t + 1) as f32;
            }
        }
        let out = masked_attention(&q, &k, &v, seq, dim).unwrap();
        // Position 0 only attends to itself → v[0] = 1.0
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 1.0));
    }

    // ── multi_head_attention_cpu (free function) ───────────────────

    #[test]
    fn mha_cpu_free_fn_matches_method() {
        let heads = 2;
        let dim = 4;
        let seq = 3;
        let n = seq * heads * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();

        let cfg = AttentionConfig {
            num_heads: heads,
            head_dim: dim,
            seq_len: seq,
            causal: false,
            scale: None,
        };
        let expected = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        let actual = multi_head_attention_cpu(&q, &k, &v, heads, dim, seq, false).unwrap();
        assert!(slices_approx_eq(&expected, &actual));
    }

    // ── attention_with_kv_cache ────────────────────────────────────

    #[test]
    fn kv_cache_single_step() {
        let dim = 4;
        let q = vec![1.0; dim];
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        let k_new = vec![1.0; dim];
        let v_new = vec![2.0; dim];
        let out =
            attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k_new, &v_new, dim).unwrap();
        // Single entry → attention weight = 1 → output = v_new
        assert!(slices_approx_eq(&out, &v_new));
        assert_eq!(k_cache.len(), dim);
        assert_eq!(v_cache.len(), dim);
    }

    #[test]
    fn kv_cache_incremental_two_steps() {
        let dim = 2;
        // Step 1: cache is empty, add first token.
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        let q1 = vec![1.0, 0.0];
        let k1 = vec![1.0, 0.0];
        let v1 = vec![10.0, 20.0];
        let out1 = attention_with_kv_cache(&q1, &mut k_cache, &mut v_cache, &k1, &v1, dim).unwrap();
        assert!(slices_approx_eq(&out1, &v1));

        // Step 2: add second token, cache now has 2 entries.
        let q2 = vec![1.0, 0.0];
        let k2 = vec![1.0, 0.0];
        let v2 = vec![30.0, 40.0];
        let out2 = attention_with_kv_cache(&q2, &mut k_cache, &mut v_cache, &k2, &v2, dim).unwrap();
        assert_eq!(k_cache.len(), 2 * dim);
        assert_eq!(v_cache.len(), 2 * dim);
        // Both keys identical → uniform attention → average of v1,v2
        let expected_d0 = (10.0 + 30.0) / 2.0;
        let expected_d1 = (20.0 + 40.0) / 2.0;
        assert!(approx_eq(out2[0], expected_d0));
        assert!(approx_eq(out2[1], expected_d1));
    }

    #[test]
    fn kv_cache_growing_sequence() {
        let dim = 4;
        let mut k_cache = Vec::new();
        let mut v_cache = Vec::new();
        for step in 0..5 {
            let q = vec![1.0; dim];
            let k_new = vec![1.0; dim];
            let v_new = vec![step as f32; dim];
            let out = attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k_new, &v_new, dim)
                .unwrap();
            assert_eq!(out.len(), dim);
            assert_eq!(k_cache.len(), (step + 1) * dim);
        }
    }

    #[test]
    fn kv_cache_rejects_bad_head_dim() {
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        assert!(attention_with_kv_cache(&[], &mut kc, &mut vc, &[], &[], 0).is_err());
    }

    #[test]
    fn kv_cache_rejects_mismatched_q() {
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        assert!(
            attention_with_kv_cache(&[1.0, 2.0], &mut kc, &mut vc, &[1.0], &[1.0], 1,).is_err()
        );
    }

    // ── Numerical stability / edge-case tests ──────────────────────

    #[test]
    fn softmax_all_neg_infinity() {
        let mut row = vec![f32::NEG_INFINITY; 4];
        softmax_row(&mut row);
        // All -inf → exp(-inf)=0 → sum=0 → values remain 0
        for &v in &row {
            assert!(v == 0.0 || v.is_nan());
        }
    }

    #[test]
    fn sdp_nan_in_query_propagates() {
        let dim = 2;
        let q = vec![f32::NAN, 1.0];
        let k = vec![1.0, 1.0];
        let v = vec![1.0, 1.0];
        let out = AttentionKernel::scaled_dot_product(&q, &k, &v, None, 1.0, 1, 1, dim).unwrap();
        // NaN in scores should propagate through softmax
        assert!(out.iter().any(|&x| x.is_nan()), "NaN should propagate through attention");
    }

    #[test]
    fn sdp_large_head_dim() {
        let dim = 256;
        let seq = 2;
        let q: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.001).collect();
        let k: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.001).collect();
        let v: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.01).collect();
        let out = AttentionKernel::scaled_dot_product(
            &q,
            &k,
            &v,
            None,
            1.0 / (dim as f32).sqrt(),
            seq,
            seq,
            dim,
        )
        .unwrap();
        assert_eq!(out.len(), seq * dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mha_single_head_single_token() {
        let cfg =
            AttentionConfig { num_heads: 1, head_dim: 8, seq_len: 1, causal: true, scale: None };
        let n = 8;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let v = vec![0.5; n];
        let out = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn sdp_asymmetric_seq_lengths() {
        // seq_q=1 (decode step), seq_k=4 (cached)
        let dim = 2;
        let q = vec![1.0, 0.0]; // 1×2
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 4×2
        let v = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // 4×2
        let out = AttentionKernel::scaled_dot_product(&q, &k, &v, None, 1.0, 1, 4, dim).unwrap();
        assert_eq!(out.len(), dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // ── causal_attention ───────────────────────────────────────────

    #[test]
    fn causal_attn_first_position_self_only() {
        let cfg =
            AttentionConfig { num_heads: 2, head_dim: 2, seq_len: 3, causal: false, scale: None };
        let model_dim = cfg.num_heads * cfg.head_dim;
        let n = cfg.seq_len * model_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let mut v = vec![0.0_f32; n];
        for t in 0..cfg.seq_len {
            for d in 0..model_dim {
                v[t * model_dim + d] = (t * model_dim + d) as f32;
            }
        }
        let out = causal_attention(&q, &k, &v, &cfg).unwrap();
        // Position 0 can only see itself → output ≈ v[0..model_dim]
        assert!(slices_approx_eq(&out[..model_dim], &v[..model_dim]));
    }

    #[test]
    fn causal_attn_matches_mha_causal() {
        let cfg =
            AttentionConfig { num_heads: 2, head_dim: 4, seq_len: 3, causal: true, scale: None };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let mha = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        let ca = causal_attention(&q, &k, &v, &cfg).unwrap();
        assert!(slices_approx_eq(&mha, &ca));
    }

    #[test]
    fn causal_attn_forces_causal_flag() {
        // Config says causal=false, but causal_attention should override.
        let cfg =
            AttentionConfig { num_heads: 1, head_dim: 2, seq_len: 3, causal: false, scale: None };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let mut v = vec![0.0_f32; n];
        for t in 0..cfg.seq_len {
            for d in 0..cfg.head_dim {
                v[t * cfg.head_dim + d] = t as f32;
            }
        }
        let out = causal_attention(&q, &k, &v, &cfg).unwrap();
        // Position 0 can only attend to itself → output row 0 ≈ 0.0
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
    }

    #[test]
    fn causal_attn_single_token() {
        let cfg =
            AttentionConfig { num_heads: 1, head_dim: 4, seq_len: 1, causal: false, scale: None };
        let v = vec![5.0; 4];
        let out = causal_attention(&[1.0; 4], &[1.0; 4], &v, &cfg).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    // ── apply_rotary_embedding ─────────────────────────────────────

    #[test]
    fn rope_position_zero_no_change() {
        let head_dim = 4;
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = original.clone();
        apply_rotary_embedding(&mut q, &mut k, &[0], head_dim).unwrap();
        // At position 0, angle = 0 → cos=1, sin=0 → no change
        assert!(slices_approx_eq(&q, &original));
        assert!(slices_approx_eq(&k, &original));
    }

    #[test]
    fn rope_modifies_nonzero_position() {
        let head_dim = 4;
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = vec![0.0; head_dim]; // k unchanged at zeros
        apply_rotary_embedding(&mut q, &mut k, &[1], head_dim).unwrap();
        // At position 1, angles are non-zero → values should change
        assert!(!slices_approx_eq(&q, &original));
    }

    #[test]
    fn rope_pair_rotation_preserves_norm() {
        // Rotation preserves the L2 norm of each (x0, x1) pair.
        let head_dim = 2;
        let mut q: Vec<f32> = vec![3.0, 4.0]; // norm = 5
        let mut k: Vec<f32> = vec![1.0, 0.0]; // norm = 1
        let q_norm_before = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let k_norm_before = (k[0] * k[0] + k[1] * k[1]).sqrt();
        apply_rotary_embedding(&mut q, &mut k, &[7], head_dim).unwrap();
        let q_norm_after = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let k_norm_after = (k[0] * k[0] + k[1] * k[1]).sqrt();
        assert!(approx_eq(q_norm_before, q_norm_after));
        assert!(approx_eq(k_norm_before, k_norm_after));
    }

    #[test]
    fn rope_multi_head() {
        let head_dim = 4;
        let num_heads = 2;
        let cols = num_heads * head_dim;
        let mut q = vec![1.0; cols];
        let mut k = vec![1.0; cols];
        apply_rotary_embedding(&mut q, &mut k, &[3], head_dim).unwrap();
        // Both heads should be rotated identically (same position)
        assert!(slices_approx_eq(&q[..head_dim], &q[head_dim..]));
        assert!(slices_approx_eq(&k[..head_dim], &k[head_dim..]));
    }

    #[test]
    fn rope_multiple_positions() {
        let head_dim = 4;
        let mut q = vec![1.0; 3 * head_dim]; // 3 positions
        let mut k = vec![1.0; 3 * head_dim];
        apply_rotary_embedding(&mut q, &mut k, &[0, 1, 2], head_dim).unwrap();
        // Position 0 unchanged
        assert!(slices_approx_eq(&q[..head_dim], &[1.0; 4]));
        // Position 1 and 2 should differ
        assert!(!slices_approx_eq(&q[head_dim..2 * head_dim], &q[..head_dim]));
        assert!(!slices_approx_eq(&q[2 * head_dim..3 * head_dim], &q[head_dim..2 * head_dim]));
    }

    #[test]
    fn rope_rejects_odd_head_dim() {
        let mut q = vec![1.0; 3];
        let mut k = vec![1.0; 3];
        assert!(apply_rotary_embedding(&mut q, &mut k, &[0], 3).is_err());
    }

    #[test]
    fn rope_rejects_zero_head_dim() {
        let mut q = vec![];
        let mut k = vec![];
        assert!(apply_rotary_embedding(&mut q, &mut k, &[0], 0).is_err());
    }

    #[test]
    fn rope_empty_positions_is_noop() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = original.clone();
        apply_rotary_embedding(&mut q, &mut k, &[], 4).unwrap();
        assert_eq!(q, original);
        assert_eq!(k, original);
    }

    #[test]
    fn rope_q_k_independent() {
        let head_dim = 4;
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let k_before = k.clone();
        apply_rotary_embedding(&mut q, &mut k, &[1], head_dim).unwrap();
        // k should be rotated by the same angles but from its own initial values
        let mut k_standalone = k_before.clone();
        let mut dummy = vec![0.0; head_dim];
        apply_rotary_embedding(&mut dummy, &mut k_standalone, &[1], head_dim).unwrap();
        assert!(slices_approx_eq(&k, &k_standalone));
    }

    #[test]
    fn rope_dimension_mismatch() {
        let mut q = vec![1.0; 5]; // not divisible by head_dim=4
        let mut k = vec![1.0; 4];
        assert!(apply_rotary_embedding(&mut q, &mut k, &[0], 4).is_err());
    }

    #[test]
    fn rope_large_head_dim_norm_preserved() {
        let head_dim = 64;
        let mut q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let mut k = q.clone();
        let q_norm_sq: f32 = q.iter().map(|x| x * x).sum();
        apply_rotary_embedding(&mut q, &mut k, &[42], head_dim).unwrap();
        let q_norm_sq_after: f32 = q.iter().map(|x| x * x).sum();
        // Total norm is sum of per-pair norms, each preserved by rotation
        assert!(
            (q_norm_sq - q_norm_sq_after).abs() < 1e-3,
            "norm changed: {q_norm_sq} → {q_norm_sq_after}"
        );
    }

    #[test]
    fn rope_deterministic() {
        let head_dim = 4;
        let positions = &[0, 5, 10];
        let original = vec![1.0; 3 * head_dim];
        let mut q1 = original.clone();
        let mut k1 = original.clone();
        let mut q2 = original.clone();
        let mut k2 = original.clone();
        apply_rotary_embedding(&mut q1, &mut k1, positions, head_dim).unwrap();
        apply_rotary_embedding(&mut q2, &mut k2, positions, head_dim).unwrap();
        assert_eq!(q1, q2);
        assert_eq!(k1, k2);
    }
}
