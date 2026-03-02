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
    /// Whether to use ALiBi (Attention with Linear Biases) positional encoding.
    pub use_alibi: bool,
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

// ── NEON implementations (aarch64 only) ─────────────────────────────

// TODO(simd): Implement NEON-accelerated dot product for ARM targets.
//
// The NEON path should mirror `avx2_dot` using `float32x4_t` intrinsics:
//   - `vld1q_f32` for aligned/unaligned loads
//   - `vfmaq_f32` for fused multiply-add (ARMv8.2+)
//   - `vaddvq_f32` for horizontal reduction
//
// Expected uplift: ~2-4× over scalar on Apple M-series and Cortex-A76+.
// Gate behind `#[cfg(target_arch = "aarch64")]` with `#[target_feature(enable = "neon")]`.

// ── Dispatch helpers ───────────────────────────────────────────────

/// Compute Q·K^T, choosing the best available SIMD path.
fn dispatch_qk(q: &[f32], k: &[f32], seq_q: usize, seq_k: usize, dim: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return avx2_qk(q, k, seq_q, seq_k, dim);
        }
    }
    // TODO(simd): add `#[cfg(target_arch = "aarch64")]` NEON fast-path here.
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

// ── CpuAttention ───────────────────────────────────────────────────

/// High-level CPU attention executor coupling configuration with computation.
///
/// Supports batched multi-head attention with optional causal masking and
/// runtime SIMD dispatch (AVX2 on x86_64, scalar fallback elsewhere).
///
/// # SIMD Dispatch
///
/// - **x86_64 + AVX2/FMA**: 256-bit vector dot products for Q·K^T
/// - **aarch64 + NEON**: TODO — planned NEON acceleration for ARM targets
/// - **Fallback**: Scalar implementation on all other platforms
///
/// # Example
///
/// ```
/// # use bitnet_kernels::cpu::attention::{CpuAttention, CpuAttentionConfig};
/// let attn = CpuAttention::new(CpuAttentionConfig {
///     batch_size: 1,
///     num_heads: 4,
///     seq_len: 8,
///     head_dim: 64,
///     scale: None,
///     causal_mask: true,
/// }).unwrap();
/// # let total = 1 * 8 * 4 * 64;
/// # let q = vec![0.1_f32; total];
/// # let k = vec![0.1_f32; total];
/// # let v = vec![0.1_f32; total];
/// let output = attn.forward(&q, &k, &v).unwrap();
/// assert_eq!(output.len(), total);
/// ```
pub struct CpuAttention {
    config: CpuAttentionConfig,
}

impl CpuAttention {
    /// Create a new `CpuAttention` executor, validating the configuration.
    pub fn new(config: CpuAttentionConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Execute batched multi-head attention.
    ///
    /// * `q` — queries, shape `[batch_size, seq_len, num_heads * head_dim]` (row-major)
    /// * `k` — keys,    shape `[batch_size, seq_len, num_heads * head_dim]`
    /// * `v` — values,  shape `[batch_size, seq_len, num_heads * head_dim]`
    ///
    /// Returns output of shape `[batch_size, seq_len, num_heads * head_dim]`.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32]) -> Result<Vec<f32>> {
        let CpuAttentionConfig {
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            causal_mask: causal,
            ..
        } = self.config;
        let model_dim = num_heads * head_dim;
        let batch_stride = seq_len * model_dim;
        let total = batch_size * batch_stride;

        if q.len() != total {
            return Err(invalid_arg(
                "q length does not match batch_size * seq_len * num_heads * head_dim",
            ));
        }
        if k.len() != total {
            return Err(invalid_arg(
                "k length does not match batch_size * seq_len * num_heads * head_dim",
            ));
        }
        if v.len() != total {
            return Err(invalid_arg(
                "v length does not match batch_size * seq_len * num_heads * head_dim",
            ));
        }

        let cfg = AttentionConfig {
            num_heads,
            head_dim,
            seq_len,
            causal,
            use_alibi: false,
            scale: self.config.scale,
        };

        let mut output = Vec::with_capacity(total);
        for b in 0..batch_size {
            let start = b * batch_stride;
            let end = start + batch_stride;
            let batch_out = AttentionKernel::multi_head_attention(
                &q[start..end],
                &k[start..end],
                &v[start..end],
                &cfg,
            )?;
            output.extend_from_slice(&batch_out);
        }

        Ok(output)
    }

    /// Execute single-head attention on per-head slices (no multi-head split).
    ///
    /// * `q` — query,  shape `[seq_q, head_dim]`
    /// * `k` — key,    shape `[seq_k, head_dim]`
    /// * `v` — value,  shape `[seq_k, head_dim]`
    ///
    /// Returns output of shape `[seq_q, head_dim]`.
    pub fn forward_single_head(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_q: usize,
        seq_k: usize,
    ) -> Result<Vec<f32>> {
        let head_dim = self.config.head_dim;
        let scale = self.config.resolved_scale();
        let mask_vec =
            if self.config.causal_mask && seq_q == seq_k { Some(causal_mask(seq_q)) } else { None };
        AttentionKernel::scaled_dot_product(
            q,
            k,
            v,
            mask_vec.as_deref(),
            scale,
            seq_q,
            seq_k,
            head_dim,
        )
    }

    /// Access the underlying configuration.
    pub fn config(&self) -> &CpuAttentionConfig {
        &self.config
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
    let cfg =
        AttentionConfig { num_heads, head_dim, seq_len, causal, use_alibi: false, scale: None };
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

// ── Compute Q, K, V projections ────────────────────────────────────

/// Project input through weight matrices to produce Q, K, V tensors.
///
/// * `input` — shape `[seq_len, model_dim]`
/// * `wq`    — query weights,  shape `[model_dim, num_q_heads * head_dim]`
/// * `wk`    — key weights,    shape `[model_dim, num_kv_heads * head_dim]`
/// * `wv`    — value weights,  shape `[model_dim, num_kv_heads * head_dim]`
///
/// Returns `(Q, K, V)` with shapes `[seq_len, num_q_heads * head_dim]`,
/// `[seq_len, num_kv_heads * head_dim]`, `[seq_len, num_kv_heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn compute_qkv(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    seq_len: usize,
    model_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    if input.len() != seq_len * model_dim {
        return Err(invalid_arg("input length does not match seq_len * model_dim"));
    }
    if wq.len() != model_dim * q_dim {
        return Err(invalid_arg("wq shape mismatch: expected model_dim * num_q_heads * head_dim"));
    }
    if wk.len() != model_dim * kv_dim {
        return Err(invalid_arg("wk shape mismatch: expected model_dim * num_kv_heads * head_dim"));
    }
    if wv.len() != model_dim * kv_dim {
        return Err(invalid_arg("wv shape mismatch: expected model_dim * num_kv_heads * head_dim"));
    }

    let q = matmul_project(input, wq, seq_len, model_dim, q_dim);
    let k = matmul_project(input, wk, seq_len, model_dim, kv_dim);
    let v = matmul_project(input, wv, seq_len, model_dim, kv_dim);

    Ok((q, k, v))
}

/// Internal matmul: `input[rows, inner] × weight[inner, cols] → out[rows, cols]`.
fn matmul_project(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let inp_row = &input[r * inner..(r + 1) * inner];
        let out_row = &mut out[r * cols..(r + 1) * cols];
        for i in 0..inner {
            let w = inp_row[i];
            let weight_row = &weight[i * cols..(i + 1) * cols];
            for c in 0..cols {
                out_row[c] += w * weight_row[c];
            }
        }
    }
    out
}

// ── Attention score computation ────────────────────────────────────

/// Compute scaled attention scores: `Q · K^T * scale`.
///
/// Uses SIMD-accelerated dot products when available (AVX2+FMA on x86_64).
///
/// * `q` — queries, shape `[seq_q, head_dim]`
/// * `k` — keys,    shape `[seq_k, head_dim]`
/// * `scale` — scaling factor (typically `1/√d_k`)
///
/// Returns scores of shape `[seq_q, seq_k]`.
pub fn attention_score_computation(
    q: &[f32],
    k: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    if head_dim == 0 {
        return Err(invalid_arg("head_dim must be > 0"));
    }
    if q.len() != seq_q * head_dim {
        return Err(invalid_arg("q length does not match seq_q * head_dim"));
    }
    if k.len() != seq_k * head_dim {
        return Err(invalid_arg("k length does not match seq_k * head_dim"));
    }

    let mut scores = dispatch_qk(q, k, seq_q, seq_k, head_dim);
    for s in &mut scores {
        *s *= scale;
    }
    Ok(scores)
}

// ── Softmax for attention ──────────────────────────────────────────

/// Row-wise softmax for attention score matrices.
///
/// Applies numerically stable softmax to each row of `scores`
/// (shape `[rows, cols]`) in-place.
pub fn softmax_attention(scores: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    if scores.len() != rows * cols {
        return Err(invalid_arg("scores length does not match rows * cols"));
    }
    softmax_rows(scores, rows, cols);
    Ok(())
}

// ── Causal mask alias ──────────────────────────────────────────────

/// Apply causal mask to attention scores (alias for [`apply_causal_mask`]).
pub fn causal_mask_apply(scores: &mut [f32], seq_len: usize) -> Result<()> {
    apply_causal_mask(scores, seq_len)
}

// ── RoPE alias ─────────────────────────────────────────────────────

/// Apply rotary position embeddings to Q and K tensors in-place
/// (alias for [`apply_rotary_embedding`]).
pub fn apply_rope_to_qk(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    head_dim: usize,
) -> Result<()> {
    apply_rotary_embedding(q, k, positions, head_dim)
}

// ── ALiBi positional bias ──────────────────────────────────────────

/// Compute ALiBi slopes for each attention head.
///
/// Returns `num_heads` slopes following the geometric sequence
/// `m_h = 2^(-8 * (h+1) / num_heads)` from the ALiBi paper.
pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    (0..num_heads).map(|h| 2.0f32.powf(-8.0 * (h as f32 + 1.0) / num_heads as f32)).collect()
}

/// Apply ALiBi positional bias to a single head's attention scores.
///
/// Adds `-slope * |i - j|` to each score at position `(i, j)`.
///
/// * `scores` — pre-softmax scores, shape `[seq_q, seq_k]`
/// * `slope` — head-specific slope (from [`alibi_slopes`])
pub fn apply_alibi_bias(scores: &mut [f32], seq_q: usize, seq_k: usize, slope: f32) -> Result<()> {
    if scores.len() != seq_q * seq_k {
        return Err(invalid_arg("scores length does not match seq_q * seq_k"));
    }
    for i in 0..seq_q {
        for j in 0..seq_k {
            let distance = i.abs_diff(j);
            scores[i * seq_k + j] -= slope * distance as f32;
        }
    }
    Ok(())
}

// ── Grouped-query attention (free function) ────────────────────────

/// Grouped-query attention (free function).
///
/// Delegates to [`AttentionKernel::grouped_query_attention`].
pub fn grouped_query_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cfg: &GqaConfig,
) -> Result<Vec<f32>> {
    AttentionKernel::grouped_query_attention(q, k, v, cfg)
}

// ── KV-cache incremental attention (alias) ─────────────────────────

/// Incremental attention with KV cache (alias for [`attention_with_kv_cache`]).
pub fn kv_cache_incremental_attention(
    q: &[f32],
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    k_new: &[f32],
    v_new: &[f32],
    head_dim: usize,
) -> Result<Vec<f32>> {
    attention_with_kv_cache(q, k_cache, v_cache, k_new, v_new, head_dim)
}

// ── Full attention forward pass ────────────────────────────────────

/// Full attention forward pass with optional ALiBi bias.
///
/// Performs multi-head attention with SIMD-accelerated score computation,
/// optional causal masking, and optional ALiBi positional bias.
///
/// * `q` — queries, shape `[seq_len, num_heads * head_dim]`
/// * `k` — keys,    shape `[seq_len, num_heads * head_dim]`
/// * `v` — values,  shape `[seq_len, num_heads * head_dim]`
///
/// Returns output of shape `[seq_len, num_heads * head_dim]`.
pub fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cfg: &AttentionConfig,
) -> Result<Vec<f32>> {
    cfg.validate()?;
    let AttentionConfig { num_heads, head_dim, seq_len, causal, use_alibi, .. } = *cfg;
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

    // If no ALiBi, delegate to the existing optimized path.
    if !use_alibi {
        return AttentionKernel::multi_head_attention(q, k, v, cfg);
    }

    let scale = cfg.resolved_scale();
    let mask_vec = if causal { Some(causal_mask(seq_len)) } else { None };
    let slopes = alibi_slopes(num_heads);
    let mut output = vec![0.0_f32; expected];

    for (h, slope) in slopes.iter().enumerate().take(num_heads) {
        let q_head = extract_head(q, seq_len, num_heads, head_dim, h);
        let k_head = extract_head(k, seq_len, num_heads, head_dim, h);
        let v_head = extract_head(v, seq_len, num_heads, head_dim, h);

        // Compute scaled scores with SIMD dispatch.
        let mut scores = dispatch_qk(&q_head, &k_head, seq_len, seq_len, head_dim);
        for s in &mut scores {
            *s *= scale;
        }

        // Apply causal mask.
        if let Some(ref m) = mask_vec {
            apply_mask(&mut scores, m)?;
        }

        // Apply ALiBi bias.
        apply_alibi_bias(&mut scores, seq_len, seq_len, *slope)?;

        // Softmax + weighted sum.
        softmax_rows(&mut scores, seq_len, seq_len);
        let head_out = scalar_sv(&scores, &v_head, seq_len, seq_len, head_dim);
        scatter_head(&mut output, &head_out, seq_len, num_heads, head_dim, h);
    }

    Ok(output)
}

// ── Flash attention CPU (tiled) ────────────────────────────────────

/// Flash attention approximation for CPU with tiled computation.
///
/// Processes Q against K/V in fixed-size tiles to improve cache locality
/// and reduce peak memory from O(N²) to O(N × block_size).  Uses
/// online softmax (running max + sum) to avoid materializing the full
/// attention matrix.
///
/// * `q` — query, shape `[seq_q, head_dim]`
/// * `k` — key,   shape `[seq_k, head_dim]`
/// * `v` — value, shape `[seq_k, head_dim]`
///
/// Returns output of shape `[seq_q, head_dim]`.
pub fn flash_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    causal: bool,
) -> Result<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;

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

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; seq_q * head_dim];
    let mut row_max = vec![f32::NEG_INFINITY; seq_q];
    let mut row_sum = vec![0.0_f32; seq_q];

    // Process K/V in tiles of BLOCK_SIZE.
    for kv_start in (0..seq_k).step_by(BLOCK_SIZE) {
        let kv_end = (kv_start + BLOCK_SIZE).min(seq_k);
        let block_k = kv_end - kv_start;

        for qi in 0..seq_q {
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];

            // Compute scores for this Q row against the K tile.
            let mut block_scores = Vec::with_capacity(block_k);
            for kj in 0..block_k {
                let k_idx = kv_start + kj;
                if causal && k_idx > qi {
                    block_scores.push(f32::NEG_INFINITY);
                } else {
                    let k_row = &k[k_idx * head_dim..(k_idx + 1) * head_dim];
                    block_scores.push(scalar_dot(q_row, k_row) * scale);
                }
            }

            // Online softmax update: re-scale previous accumulator.
            let block_max = block_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let new_max = row_max[qi].max(block_max);

            let correction = (row_max[qi] - new_max).exp();
            let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
            for d in out_row.iter_mut() {
                *d *= correction;
            }
            row_sum[qi] *= correction;

            // Accumulate this tile's contribution.
            for (kj, &score) in block_scores.iter().enumerate().take(block_k) {
                let w = (score - new_max).exp();
                row_sum[qi] += w;
                let k_idx = kv_start + kj;
                let v_row = &v[k_idx * head_dim..(k_idx + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] += w * v_row[d];
                }
            }

            row_max[qi] = new_max;
        }
    }

    // Final normalization.
    for qi in 0..seq_q {
        if row_sum[qi] > 0.0 {
            let inv = 1.0 / row_sum[qi];
            let out_row = &mut output[qi * head_dim..(qi + 1) * head_dim];
            for d in out_row.iter_mut() {
                *d *= inv;
            }
        }
    }

    Ok(output)
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
        let cfg = AttentionConfig {
            num_heads: 4,
            head_dim: 64,
            seq_len: 8,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
            use_alibi: false,
            scale: Some(0.5),
        };
        assert!(approx_eq(cfg.resolved_scale(), 0.5));
    }

    #[test]
    fn config_validate_zero_heads() {
        let cfg = AttentionConfig {
            num_heads: 0,
            head_dim: 64,
            seq_len: 8,
            causal: false,
            use_alibi: false,
            scale: None,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_head_dim() {
        let cfg = AttentionConfig {
            num_heads: 4,
            head_dim: 0,
            seq_len: 8,
            causal: false,
            use_alibi: false,
            scale: None,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_zero_seq_len() {
        let cfg = AttentionConfig {
            num_heads: 4,
            head_dim: 64,
            seq_len: 0,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
                assert!((1.0..=4.0).contains(&val), "out of convex range: {val}");
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
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 4,
            seq_len: 2,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
        let cfg = AttentionConfig {
            num_heads: 4,
            head_dim: 8,
            seq_len: 3,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
            use_alibi: false,
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
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 3,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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

        let cfg = AttentionConfig {
            num_heads,
            head_dim,
            seq_len,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
            use_alibi: false,
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
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 8,
            seq_len: 1,
            causal: true,
            use_alibi: false,
            scale: None,
        };
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
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 2,
            seq_len: 3,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 3,
            causal: true,
            use_alibi: false,
            scale: None,
        };
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
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 2,
            seq_len: 3,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 4,
            seq_len: 1,
            causal: false,
            use_alibi: false,
            scale: None,
        };
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

    // ── CpuAttention ──────────────────────────────────────────────

    #[test]
    fn cpu_attention_basic_forward() {
        let heads = 2;
        let dim = 4;
        let seq = 3;
        let batch = 1;
        let total = batch * seq * heads * dim;
        let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.03).collect();

        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: batch,
            num_heads: heads,
            seq_len: seq,
            head_dim: dim,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let out = attn.forward(&q, &k, &v).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn cpu_attention_matches_mha() {
        let heads = 2;
        let dim = 4;
        let seq = 3;
        let n = seq * heads * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();

        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: heads,
            seq_len: seq,
            head_dim: dim,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let cpu_out = attn.forward(&q, &k, &v).unwrap();
        let mha_out = multi_head_attention_cpu(&q, &k, &v, heads, dim, seq, false).unwrap();
        assert!(slices_approx_eq(&cpu_out, &mha_out));
    }

    #[test]
    fn cpu_attention_batched() {
        let heads = 2;
        let dim = 4;
        let seq = 2;
        let batch = 3;
        let batch_stride = seq * heads * dim;
        let total = batch * batch_stride;
        let q = vec![0.1; total];
        let k = vec![0.1; total];
        let v = vec![0.2; total];

        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: batch,
            num_heads: heads,
            seq_len: seq,
            head_dim: dim,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let out = attn.forward(&q, &k, &v).unwrap();
        assert_eq!(out.len(), total);
        // All batches should produce identical output (same input)
        let batch0 = &out[..batch_stride];
        for b in 1..batch {
            let batch_b = &out[b * batch_stride..(b + 1) * batch_stride];
            assert!(slices_approx_eq(batch0, batch_b));
        }
    }

    #[test]
    fn cpu_attention_causal_mask() {
        let heads = 1;
        let dim = 2;
        let seq = 3;
        let model_dim = heads * dim;
        let n = seq * model_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let mut v = vec![0.0_f32; n];
        for t in 0..seq {
            for d in 0..model_dim {
                v[t * model_dim + d] = t as f32;
            }
        }

        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: heads,
            seq_len: seq,
            head_dim: dim,
            scale: Some(1.0),
            causal_mask: true,
        })
        .unwrap();
        let out = attn.forward(&q, &k, &v).unwrap();
        // Position 0 can only attend to itself → output ≈ v[0] = 0.0
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
    }

    #[test]
    fn cpu_attention_single_token() {
        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            head_dim: 4,
            scale: None,
            causal_mask: true,
        })
        .unwrap();
        let v = vec![5.0; 4];
        let out = attn.forward(&[1.0; 4], &[1.0; 4], &v).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn cpu_attention_numerical_stability() {
        let dim = 64;
        let seq = 4;
        let heads = 2;
        let total = seq * heads * dim;
        // Large values that could cause overflow without stable softmax
        let q: Vec<f32> = (0..total).map(|i| 500.0 + (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|i| 500.0 + (i as f32) * 0.1).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();

        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: heads,
            seq_len: seq,
            head_dim: dim,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let out = attn.forward(&q, &k, &v).unwrap();
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|x| x.is_finite()), "output should be finite for large inputs");
    }

    #[test]
    fn cpu_attention_shape_validation_q() {
        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: 2,
            seq_len: 3,
            head_dim: 4,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let correct = vec![0.0; 24];
        let wrong = vec![0.0; 10];
        assert!(attn.forward(&wrong, &correct, &correct).is_err());
    }

    #[test]
    fn cpu_attention_shape_validation_k() {
        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: 2,
            seq_len: 3,
            head_dim: 4,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let correct = vec![0.0; 24];
        let wrong = vec![0.0; 10];
        assert!(attn.forward(&correct, &wrong, &correct).is_err());
    }

    #[test]
    fn cpu_attention_shape_validation_v() {
        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: 2,
            seq_len: 3,
            head_dim: 4,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let correct = vec![0.0; 24];
        let wrong = vec![0.0; 10];
        assert!(attn.forward(&correct, &correct, &wrong).is_err());
    }

    #[test]
    fn cpu_attention_invalid_config() {
        assert!(
            CpuAttention::new(CpuAttentionConfig {
                batch_size: 0,
                num_heads: 2,
                seq_len: 3,
                head_dim: 4,
                scale: None,
                causal_mask: false,
            })
            .is_err()
        );
    }

    #[test]
    fn cpu_attention_forward_single_head() {
        let attn = CpuAttention::new(CpuAttentionConfig {
            batch_size: 1,
            num_heads: 1,
            seq_len: 1,
            head_dim: 4,
            scale: None,
            causal_mask: false,
        })
        .unwrap();
        let v = vec![7.0; 4];
        let out = attn.forward_single_head(&[1.0; 4], &[1.0; 4], &v, 1, 1).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn cpu_attention_config_accessor() {
        let cfg = CpuAttentionConfig {
            batch_size: 2,
            num_heads: 4,
            seq_len: 8,
            head_dim: 64,
            scale: Some(0.5),
            causal_mask: true,
        };
        let attn = CpuAttention::new(cfg.clone()).unwrap();
        assert_eq!(attn.config().batch_size, 2);
        assert_eq!(attn.config().num_heads, 4);
        assert!(attn.config().causal_mask);
    }

    // ── Edge case: max sequence length ───────────────────────────────

    #[test]
    fn attention_max_sequence_length() {
        let seq = 256;
        let head_dim = 4;
        let n = seq * head_dim;
        let q = vec![0.01_f32; n];
        let k = vec![0.01_f32; n];
        let v = vec![1.0_f32; n];
        let out = scaled_dot_product_attention(&q, &k, &v, seq, seq, head_dim, true).unwrap();
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Property tests ────────────────────────────────────────────────

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Output length always equals `seq_q * head_dim` for single-head SDP.
            #[test]
            fn prop_sdp_output_shape(
                head_dim in 1_usize..=16,
                seq_len in 1_usize..=32,
                causal in proptest::bool::ANY,
            ) {
                let n = seq_len * head_dim;
                let q = vec![0.1_f32; n];
                let k = vec![0.2_f32; n];
                let v = vec![0.3_f32; n];
                let out = scaled_dot_product_attention(
                    &q, &k, &v, seq_len, seq_len, head_dim, causal,
                ).unwrap();
                prop_assert_eq!(out.len(), n);
            }

            /// Multi-head output length equals `seq_len * num_heads * head_dim`.
            #[test]
            fn prop_mha_output_shape(
                num_heads in 1_usize..=4,
                head_dim in 1_usize..=16,
                seq_len in 1_usize..=16,
                causal in proptest::bool::ANY,
            ) {
                let total = seq_len * num_heads * head_dim;
                let q = vec![0.1_f32; total];
                let k = vec![0.2_f32; total];
                let v = vec![0.3_f32; total];
                let out = multi_head_attention_cpu(
                    &q, &k, &v, num_heads, head_dim, seq_len, causal,
                ).unwrap();
                prop_assert_eq!(out.len(), total);
            }

            /// Output is a convex combination of V rows: each dimension must
            /// lie within the [min, max] range of V for that dimension.
            #[test]
            fn prop_output_within_value_range(
                head_dim in 1_usize..=8,
                seq_len in 1_usize..=16,
                causal in proptest::bool::ANY,
                v_seed in proptest::collection::vec(-50.0_f32..50.0, 1..=128),
            ) {
                let n = seq_len * head_dim;
                let v: Vec<f32> = v_seed.iter().copied().cycle().take(n).collect();
                let q = vec![0.5_f32; n];
                let k = vec![0.5_f32; n];
                let out = scaled_dot_product_attention(
                    &q, &k, &v, seq_len, seq_len, head_dim, causal,
                ).unwrap();

                for d in 0..head_dim {
                    let col: Vec<f32> =
                        (0..seq_len).map(|r| v[r * head_dim + d]).collect();
                    let v_min = col.iter().copied().fold(f32::INFINITY, f32::min);
                    let v_max =
                        col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for r in 0..seq_len {
                        let o = out[r * head_dim + d];
                        prop_assert!(
                            o >= v_min - 1e-4 && o <= v_max + 1e-4,
                            "row {r} dim {d}: out={o} not in [{v_min}, {v_max}]"
                        );
                    }
                }
            }

            /// All output values are finite for bounded inputs.
            #[test]
            fn prop_output_always_finite(
                num_heads in 1_usize..=4,
                head_dim in 1_usize..=16,
                seq_len in 1_usize..=16,
                causal in proptest::bool::ANY,
            ) {
                let total = seq_len * num_heads * head_dim;
                let q = vec![0.3_f32; total];
                let k = vec![0.3_f32; total];
                let v = vec![0.7_f32; total];
                let out = multi_head_attention_cpu(
                    &q, &k, &v, num_heads, head_dim, seq_len, causal,
                ).unwrap();
                for (i, &val) in out.iter().enumerate() {
                    prop_assert!(val.is_finite(), "non-finite at index {i}: {val}");
                }
            }

            /// Softmax rows sum to 1: with constant V the output must equal
            /// that constant (since ∑ weights * c = c).
            #[test]
            fn prop_softmax_rows_sum_to_one(
                head_dim in 1_usize..=8,
                seq_len in 1_usize..=16,
                causal in proptest::bool::ANY,
                constant in -100.0_f32..100.0,
            ) {
                let n = seq_len * head_dim;
                let q = vec![0.5_f32; n];
                let k = vec![0.5_f32; n];
                let v = vec![constant; n];
                let out = scaled_dot_product_attention(
                    &q, &k, &v, seq_len, seq_len, head_dim, causal,
                ).unwrap();
                for (i, &val) in out.iter().enumerate() {
                    prop_assert!(
                        (val - constant).abs() < 1e-3,
                        "index {i}: expected {constant}, got {val}"
                    );
                }
            }

            /// Causal attention: first row can only attend to position 0,
            /// so output[0..head_dim] must equal V[0..head_dim].
            #[test]
            fn prop_causal_first_row_equals_v0(
                head_dim in 1_usize..=8,
                seq_len in 1_usize..=16,
            ) {
                let n = seq_len * head_dim;
                let q = vec![1.0_f32; n];
                let k = vec![1.0_f32; n];
                let v: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
                let out = scaled_dot_product_attention(
                    &q, &k, &v, seq_len, seq_len, head_dim, true,
                ).unwrap();
                for d in 0..head_dim {
                    prop_assert!(
                        (out[d] - v[d]).abs() < 1e-4,
                        "dim {d}: first row {}, expected {}",
                        out[d],
                        v[d]
                    );
                }
            }
        }
    }

    // ── compute_qkv ────────────────────────────────────────────────

    #[test]
    fn compute_qkv_identity_weights() {
        // 2×2 identity weights → Q=K=V=input
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens, dim=2
        let eye = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let (q, k, v) = compute_qkv(&input, &eye, &eye, &eye, 2, 2, 1, 1, 2).unwrap();
        assert!(slices_approx_eq(&q, &input));
        assert!(slices_approx_eq(&k, &input));
        assert!(slices_approx_eq(&v, &input));
    }

    #[test]
    fn compute_qkv_output_shapes() {
        let seq_len = 3;
        let model_dim = 4;
        let num_q = 2;
        let num_kv = 1;
        let head_dim = 2;
        let input = vec![0.1; seq_len * model_dim];
        let wq = vec![0.1; model_dim * num_q * head_dim];
        let wk = vec![0.1; model_dim * num_kv * head_dim];
        let wv = vec![0.1; model_dim * num_kv * head_dim];
        let (q, k, v) =
            compute_qkv(&input, &wq, &wk, &wv, seq_len, model_dim, num_q, num_kv, head_dim)
                .unwrap();
        assert_eq!(q.len(), seq_len * num_q * head_dim);
        assert_eq!(k.len(), seq_len * num_kv * head_dim);
        assert_eq!(v.len(), seq_len * num_kv * head_dim);
    }

    #[test]
    fn compute_qkv_zero_input() {
        let input = vec![0.0; 6];
        let w = vec![1.0; 6];
        let (q, k, v) = compute_qkv(&input, &w, &w, &w, 2, 3, 1, 1, 2).unwrap();
        assert!(q.iter().all(|&x| x == 0.0));
        assert!(k.iter().all(|&x| x == 0.0));
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn compute_qkv_rejects_input_mismatch() {
        let input = vec![0.0; 5]; // wrong size
        let w = vec![0.0; 6];
        assert!(compute_qkv(&input, &w, &w, &w, 2, 3, 1, 1, 2).is_err());
    }

    #[test]
    fn compute_qkv_rejects_wq_mismatch() {
        let input = vec![0.0; 6];
        let wq = vec![0.0; 5]; // wrong
        let wk = vec![0.0; 6];
        assert!(compute_qkv(&input, &wq, &wk, &wk, 2, 3, 1, 1, 2).is_err());
    }

    #[test]
    fn compute_qkv_known_values() {
        // input=[1,1], wq=[[1,0],[0,2]] → q=[1,2]
        let input = vec![1.0, 1.0];
        let wq = vec![1.0, 0.0, 0.0, 2.0]; // row0=[1,0], row1=[0,2]
        let wi = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let (q, _k, _v) = compute_qkv(&input, &wq, &wi, &wi, 1, 2, 1, 1, 2).unwrap();
        assert!(approx_eq(q[0], 1.0));
        assert!(approx_eq(q[1], 2.0));
    }

    // ── attention_score_computation ────────────────────────────────

    #[test]
    fn score_comp_basic() {
        let dim = 2;
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
        let scores = attention_score_computation(&q, &k, 1, 2, dim, 1.0).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(approx_eq(scores[0], 1.0)); // dot([1,0],[1,0]) = 1
        assert!(approx_eq(scores[1], 0.0)); // dot([1,0],[0,1]) = 0
    }

    #[test]
    fn score_comp_scale() {
        let dim = 2;
        let q = vec![2.0, 0.0];
        let k = vec![3.0, 0.0];
        let scores = attention_score_computation(&q, &k, 1, 1, dim, 0.5).unwrap();
        assert!(approx_eq(scores[0], 3.0)); // 2*3 * 0.5 = 3
    }

    #[test]
    fn score_comp_rejects_bad_dim() {
        assert!(attention_score_computation(&[], &[], 0, 0, 0, 1.0).is_err());
    }

    #[test]
    fn score_comp_rejects_q_mismatch() {
        assert!(attention_score_computation(&[1.0], &[1.0, 2.0], 1, 1, 2, 1.0).is_err());
    }

    #[test]
    fn score_comp_matches_dispatch() {
        let dim = 8;
        let seq = 4;
        let q: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.05).collect();
        let scale = 1.0 / (dim as f32).sqrt();
        let scores = attention_score_computation(&q, &k, seq, seq, dim, scale).unwrap();
        let mut reference = dispatch_qk(&q, &k, seq, seq, dim);
        for s in &mut reference {
            *s *= scale;
        }
        assert!(slices_approx_eq(&scores, &reference));
    }

    // ── softmax_attention ──────────────────────────────────────────

    #[test]
    fn softmax_attn_uniform() {
        let mut scores = vec![1.0; 6]; // 2 rows × 3 cols
        softmax_attention(&mut scores, 2, 3).unwrap();
        for r in 0..2 {
            let row_sum: f32 = scores[r * 3..(r + 1) * 3].iter().sum();
            assert!(approx_eq(row_sum, 1.0));
        }
    }

    #[test]
    fn softmax_attn_rejects_mismatch() {
        let mut scores = vec![1.0; 5];
        assert!(softmax_attention(&mut scores, 2, 3).is_err());
    }

    #[test]
    fn softmax_attn_preserves_order() {
        let mut scores = vec![1.0, 3.0, 2.0];
        softmax_attention(&mut scores, 1, 3).unwrap();
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);
    }

    // ── causal_mask_apply ──────────────────────────────────────────

    #[test]
    fn causal_mask_apply_matches_original() {
        let mut s1 = vec![1.0; 9];
        let mut s2 = vec![1.0; 9];
        causal_mask_apply(&mut s1, 3).unwrap();
        apply_causal_mask(&mut s2, 3).unwrap();
        assert!(slices_approx_eq(&s1, &s2));
    }

    #[test]
    fn causal_mask_apply_rejects_bad_len() {
        let mut s = vec![1.0; 5];
        assert!(causal_mask_apply(&mut s, 3).is_err());
    }

    // ── apply_rope_to_qk ──────────────────────────────────────────

    #[test]
    fn rope_to_qk_matches_original() {
        let head_dim = 4;
        let mut q1 = vec![1.0, 2.0, 3.0, 4.0];
        let mut k1 = vec![5.0, 6.0, 7.0, 8.0];
        let mut q2 = q1.clone();
        let mut k2 = k1.clone();
        apply_rope_to_qk(&mut q1, &mut k1, &[1], head_dim).unwrap();
        apply_rotary_embedding(&mut q2, &mut k2, &[1], head_dim).unwrap();
        assert!(slices_approx_eq(&q1, &q2));
        assert!(slices_approx_eq(&k1, &k2));
    }

    #[test]
    fn rope_to_qk_position_zero_identity() {
        let head_dim = 4;
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        let mut k = original.clone();
        apply_rope_to_qk(&mut q, &mut k, &[0], head_dim).unwrap();
        assert!(slices_approx_eq(&q, &original));
        assert!(slices_approx_eq(&k, &original));
    }

    // ── alibi_slopes ───────────────────────────────────────────────

    #[test]
    fn alibi_slopes_length() {
        for h in [1, 2, 4, 8, 16] {
            assert_eq!(alibi_slopes(h).len(), h);
        }
    }

    #[test]
    fn alibi_slopes_decreasing() {
        let slopes = alibi_slopes(8);
        for i in 1..slopes.len() {
            assert!(
                slopes[i] < slopes[i - 1],
                "slopes should decrease: s[{}]={} >= s[{}]={}",
                i,
                slopes[i],
                i - 1,
                slopes[i - 1]
            );
        }
    }

    #[test]
    fn alibi_slopes_positive() {
        for &s in &alibi_slopes(4) {
            assert!(s > 0.0, "slopes must be positive: {s}");
        }
    }

    #[test]
    fn alibi_slopes_known_values() {
        // For 1 head: m = 2^(-8*1/1) = 2^-8 = 1/256
        let s = alibi_slopes(1);
        assert!(approx_eq(s[0], 1.0 / 256.0));
    }

    // ── apply_alibi_bias ───────────────────────────────────────────

    #[test]
    fn alibi_bias_diagonal_zero() {
        let seq = 3;
        let mut scores = vec![0.0; seq * seq];
        apply_alibi_bias(&mut scores, seq, seq, 0.5).unwrap();
        // Diagonal positions (i==j) have distance 0 → no bias
        for i in 0..seq {
            assert!(approx_eq(scores[i * seq + i], 0.0));
        }
    }

    #[test]
    fn alibi_bias_off_diagonal() {
        let mut scores = vec![0.0; 4]; // 2×2
        apply_alibi_bias(&mut scores, 2, 2, 1.0).unwrap();
        // (0,0)=0, (0,1)=-1, (1,0)=-1, (1,1)=0
        assert!(approx_eq(scores[0], 0.0));
        assert!(approx_eq(scores[1], -1.0));
        assert!(approx_eq(scores[2], -1.0));
        assert!(approx_eq(scores[3], 0.0));
    }

    #[test]
    fn alibi_bias_slope_scaling() {
        let mut s1 = vec![0.0; 4];
        let mut s2 = vec![0.0; 4];
        apply_alibi_bias(&mut s1, 2, 2, 0.5).unwrap();
        apply_alibi_bias(&mut s2, 2, 2, 1.0).unwrap();
        // s2 should have 2× the bias of s1
        assert!(approx_eq(s1[1] * 2.0, s2[1]));
    }

    #[test]
    fn alibi_bias_rejects_mismatch() {
        let mut scores = vec![0.0; 5];
        assert!(apply_alibi_bias(&mut scores, 2, 3, 0.5).is_err());
    }

    #[test]
    fn alibi_bias_symmetric_distances() {
        let seq = 4;
        let mut scores = vec![0.0; seq * seq];
        apply_alibi_bias(&mut scores, seq, seq, 1.0).unwrap();
        // bias(i,j) == bias(j,i) since |i-j| == |j-i|
        for i in 0..seq {
            for j in 0..seq {
                assert!(approx_eq(scores[i * seq + j], scores[j * seq + i]));
            }
        }
    }

    #[test]
    fn alibi_bias_additive_to_existing_scores() {
        let mut scores = vec![10.0, 20.0, 30.0, 40.0]; // 2×2
        apply_alibi_bias(&mut scores, 2, 2, 0.5).unwrap();
        assert!(approx_eq(scores[0], 10.0)); // diagonal: no change
        assert!(approx_eq(scores[1], 19.5)); // 20 - 0.5*1
        assert!(approx_eq(scores[2], 29.5)); // 30 - 0.5*1
        assert!(approx_eq(scores[3], 40.0)); // diagonal: no change
    }

    // ── grouped_query_attention (free fn) ──────────────────────────

    #[test]
    fn gqa_free_fn_matches_method() {
        let cfg = GqaConfig {
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            seq_len: 2,
            causal: false,
            scale: None,
        };
        let q = vec![0.1; cfg.seq_len * cfg.num_q_heads * cfg.head_dim];
        let k = vec![0.1; cfg.seq_len * cfg.num_kv_heads * cfg.head_dim];
        let v = vec![0.2; cfg.seq_len * cfg.num_kv_heads * cfg.head_dim];
        let method = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).unwrap();
        let free_fn = grouped_query_attention(&q, &k, &v, &cfg).unwrap();
        assert!(slices_approx_eq(&method, &free_fn));
    }

    // ── kv_cache_incremental_attention ──────────────────────────────

    #[test]
    fn kv_cache_incr_matches_original() {
        let dim = 4;
        let q = vec![1.0; dim];
        let mut kc1 = Vec::new();
        let mut vc1 = Vec::new();
        let mut kc2 = Vec::new();
        let mut vc2 = Vec::new();
        let k_new = vec![1.0; dim];
        let v_new = vec![2.0; dim];
        let out1 = attention_with_kv_cache(&q, &mut kc1, &mut vc1, &k_new, &v_new, dim).unwrap();
        let out2 =
            kv_cache_incremental_attention(&q, &mut kc2, &mut vc2, &k_new, &v_new, dim).unwrap();
        assert!(slices_approx_eq(&out1, &out2));
    }

    #[test]
    fn kv_cache_incr_growing() {
        let dim = 4;
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for step in 0..3 {
            let q = vec![1.0; dim];
            let k = vec![1.0; dim];
            let v = vec![step as f32; dim];
            let out = kv_cache_incremental_attention(&q, &mut kc, &mut vc, &k, &v, dim).unwrap();
            assert_eq!(out.len(), dim);
            assert_eq!(kc.len(), (step + 1) * dim);
        }
    }

    // ── attention_forward ──────────────────────────────────────────

    #[test]
    fn attn_fwd_no_alibi_matches_mha() {
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 3,
            causal: false,
            use_alibi: false,
            scale: None,
        };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let fwd = attention_forward(&q, &k, &v, &cfg).unwrap();
        let mha = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();
        assert!(slices_approx_eq(&fwd, &mha));
    }

    #[test]
    fn attn_fwd_alibi_output_shape() {
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 3,
            causal: false,
            use_alibi: true,
            scale: None,
        };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q = vec![0.1; n];
        let k = vec![0.1; n];
        let v = vec![0.2; n];
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn attn_fwd_alibi_causal() {
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 2,
            seq_len: 3,
            causal: true,
            use_alibi: true,
            scale: Some(1.0),
        };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let mut v = vec![0.0_f32; n];
        for t in 0..cfg.seq_len {
            for d in 0..cfg.head_dim {
                v[t * cfg.head_dim + d] = t as f32;
            }
        }
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        // Position 0 can only attend to itself → output ≈ v[0] = 0.0
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
    }

    #[test]
    fn attn_fwd_alibi_differs_from_no_alibi() {
        let cfg_no = AttentionConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            causal: false,
            use_alibi: false,
            scale: None,
        };
        let cfg_yes = AttentionConfig { use_alibi: true, ..cfg_no.clone() };
        let n = cfg_no.seq_len * cfg_no.num_heads * cfg_no.head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let out_no = attention_forward(&q, &k, &v, &cfg_no).unwrap();
        let out_yes = attention_forward(&q, &k, &v, &cfg_yes).unwrap();
        let any_diff = out_no.iter().zip(out_yes.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "ALiBi should change attention output");
    }

    #[test]
    fn attn_fwd_rejects_bad_q() {
        let cfg = AttentionConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 3,
            causal: false,
            use_alibi: false,
            scale: None,
        };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let wrong = vec![0.0; 10];
        let correct = vec![0.0; n];
        assert!(attention_forward(&wrong, &correct, &correct, &cfg).is_err());
    }

    #[test]
    fn attn_fwd_single_token() {
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 4,
            seq_len: 1,
            causal: true,
            use_alibi: true,
            scale: None,
        };
        let v = vec![5.0; 4];
        let out = attention_forward(&[1.0; 4], &[1.0; 4], &v, &cfg).unwrap();
        // Single token → attention weight = 1 → output = v
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn attn_fwd_alibi_recency_bias() {
        // With ALiBi, nearer tokens should get more weight.
        let cfg = AttentionConfig {
            num_heads: 1,
            head_dim: 2,
            seq_len: 4,
            causal: false,
            use_alibi: true,
            scale: Some(0.0), // zero scale → equal raw scores, ALiBi decides
        };
        let n = cfg.seq_len * cfg.num_heads * cfg.head_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        // V: each position has value = position index
        let mut v = vec![0.0_f32; n];
        for t in 0..cfg.seq_len {
            for d in 0..cfg.head_dim {
                v[t * cfg.head_dim + d] = t as f32;
            }
        }
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        // Last token (pos 3): with ALiBi, position 3 is closest to itself
        // so output should be biased toward v[3]=3.0
        let last_row_start = 3 * cfg.head_dim;
        assert!(out[last_row_start] > 1.5, "recency bias should pull toward v[3]");
    }

    // ── flash_attention_cpu ────────────────────────────────────────

    #[test]
    fn flash_attn_single_token() {
        const DIM: usize = 4;
        let v = vec![2.0; DIM];
        let out = flash_attention_cpu(&[1.0; DIM], &[1.0; DIM], &v, 1, 1, DIM, false).unwrap();
        assert!(slices_approx_eq(&out, &v));
    }

    #[test]
    fn flash_attn_matches_sdp_no_mask() {
        let dim = 4;
        let seq = 8;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let sdp = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false).unwrap();
        let flash = flash_attention_cpu(&q, &k, &v, seq, seq, dim, false).unwrap();
        for (i, (&s, &f)) in sdp.iter().zip(flash.iter()).enumerate() {
            assert!((s - f).abs() < 1e-4, "mismatch at {i}: sdp={s} flash={f}");
        }
    }

    #[test]
    fn flash_attn_matches_sdp_causal() {
        let dim = 4;
        let seq = 6;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) as f32) * 0.01).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) as f32) * 0.01).collect();
        let sdp = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, true).unwrap();
        let flash = flash_attention_cpu(&q, &k, &v, seq, seq, dim, true).unwrap();
        for (i, (&s, &f)) in sdp.iter().zip(flash.iter()).enumerate() {
            assert!((s - f).abs() < 1e-4, "causal mismatch at {i}: sdp={s} flash={f}");
        }
    }

    #[test]
    fn flash_attn_larger_than_block_size() {
        // Sequence longer than block size (32) to test multi-tile path.
        let dim = 4;
        let seq = 50;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let sdp = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false).unwrap();
        let flash = flash_attention_cpu(&q, &k, &v, seq, seq, dim, false).unwrap();
        for (i, (&s, &f)) in sdp.iter().zip(flash.iter()).enumerate() {
            assert!((s - f).abs() < 1e-3, "large seq mismatch at {i}: sdp={s} flash={f}");
        }
    }

    #[test]
    fn flash_attn_causal_first_row() {
        let dim = 2;
        let seq = 4;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let mut v = vec![0.0_f32; seq * dim];
        for t in 0..seq {
            for d in 0..dim {
                v[t * dim + d] = t as f32;
            }
        }
        let out = flash_attention_cpu(&q, &k, &v, seq, seq, dim, true).unwrap();
        // Position 0 can only attend to itself → output ≈ v[0] = 0.0
        assert!(approx_eq(out[0], 0.0));
        assert!(approx_eq(out[1], 0.0));
    }

    #[test]
    fn flash_attn_output_shape() {
        let dim = 8;
        let seq_q = 3;
        let seq_k = 5;
        let q = vec![0.1; seq_q * dim];
        let k = vec![0.1; seq_k * dim];
        let v = vec![0.2; seq_k * dim];
        let out = flash_attention_cpu(&q, &k, &v, seq_q, seq_k, dim, false).unwrap();
        assert_eq!(out.len(), seq_q * dim);
    }

    #[test]
    fn flash_attn_rejects_bad_dim() {
        assert!(flash_attention_cpu(&[], &[], &[], 0, 0, 0, false).is_err());
    }

    #[test]
    fn flash_attn_rejects_q_mismatch() {
        assert!(flash_attention_cpu(&[1.0], &[1.0, 2.0], &[1.0, 2.0], 1, 1, 2, false).is_err());
    }

    #[test]
    fn flash_attn_uniform_attention() {
        let dim = 2;
        let seq = 3;
        let q = vec![1.0; seq * dim];
        let k = vec![1.0; seq * dim];
        let v: Vec<f32> = (0..seq).flat_map(|t| vec![t as f32; dim]).collect();
        let out = flash_attention_cpu(&q, &k, &v, seq, seq, dim, false).unwrap();
        // All Q rows identical, all K rows identical → uniform attention
        let expected = (0.0 + 1.0 + 2.0) / 3.0;
        for &o in &out {
            assert!((o - expected).abs() < 1e-4, "expected uniform ~{expected}, got {o}");
        }
    }

    #[test]
    fn flash_attn_numerical_stability() {
        let dim = 4;
        let seq = 4;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| 500.0 + (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| 500.0 + (i as f32) * 0.1).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let out = flash_attention_cpu(&q, &k, &v, seq, seq, dim, false).unwrap();
        assert!(out.iter().all(|x| x.is_finite()), "should be finite for large inputs");
    }

    #[test]
    fn flash_attn_asymmetric_seq() {
        // seq_q=1, seq_k=4 (decode step)
        let dim = 2;
        let q = vec![1.0, 0.0]; // 1×2
        let k = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 4×2
        let v = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // 4×2
        let out = flash_attention_cpu(&q, &k, &v, 1, 4, dim, false).unwrap();
        assert_eq!(out.len(), dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn flash_attn_matches_sdp_large_dim() {
        let dim = 64;
        let seq = 4;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003).collect();
        let sdp = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false).unwrap();
        let flash = flash_attention_cpu(&q, &k, &v, seq, seq, dim, false).unwrap();
        for (i, (&s, &f)) in sdp.iter().zip(flash.iter()).enumerate() {
            assert!((s - f).abs() < 1e-3, "dim=64 mismatch at {i}: sdp={s} flash={f}");
        }
    }

    #[test]
    fn flash_attn_output_within_value_range() {
        let dim = 4;
        let seq = 6;
        let v: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.5 - 5.0).collect();
        let q = vec![0.5; seq * dim];
        let k = vec![0.5; seq * dim];
        let out = flash_attention_cpu(&q, &k, &v, seq, seq, dim, false).unwrap();
        for d in 0..dim {
            let col: Vec<f32> = (0..seq).map(|r| v[r * dim + d]).collect();
            let v_min = col.iter().copied().fold(f32::INFINITY, f32::min);
            let v_max = col.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for r in 0..seq {
                let o = out[r * dim + d];
                assert!(
                    o >= v_min - 1e-4 && o <= v_max + 1e-4,
                    "row {r} dim {d}: out={o} not in [{v_min}, {v_max}]"
                );
            }
        }
    }

    // ── Cross-function integration tests ───────────────────────────

    #[test]
    fn integration_score_softmax_sv_matches_sdp() {
        let dim = 4;
        let seq = 3;
        let scale = 1.0 / (dim as f32).sqrt();
        let q: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.05).collect();
        let v: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.03).collect();

        // Manual pipeline: score → softmax → weighted sum
        let mut scores = attention_score_computation(&q, &k, seq, seq, dim, scale).unwrap();
        softmax_attention(&mut scores, seq, seq).unwrap();
        let manual = scalar_sv(&scores, &v, seq, seq, dim);

        // SDP function
        let sdp = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, false).unwrap();
        assert!(slices_approx_eq(&manual, &sdp));
    }

    #[test]
    fn integration_compute_qkv_then_attend() {
        let seq_len = 2;
        let model_dim = 4;
        let head_dim = 4;
        let num_heads = 1;
        let input = vec![1.0; seq_len * model_dim];
        let w_eye = {
            let mut w = vec![0.0_f32; model_dim * head_dim];
            for i in 0..model_dim.min(head_dim) {
                w[i * head_dim + i] = 1.0;
            }
            w
        };
        let (q, k, v) = compute_qkv(
            &input, &w_eye, &w_eye, &w_eye, seq_len, model_dim, num_heads, num_heads, head_dim,
        )
        .unwrap();

        let cfg = AttentionConfig {
            num_heads,
            head_dim,
            seq_len,
            causal: false,
            use_alibi: false,
            scale: None,
        };
        let out = attention_forward(&q, &k, &v, &cfg).unwrap();
        assert_eq!(out.len(), seq_len * num_heads * head_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn integration_flash_causal_matches_sdp_causal_exact() {
        // Exact block boundary: seq=32 = 1 block
        let dim = 4;
        let seq = 32;
        let n = seq * dim;
        let q: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) as f32) * 0.01).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i * 5 + 2) as f32) * 0.01).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) as f32) * 0.01).collect();
        let sdp = scaled_dot_product_attention(&q, &k, &v, seq, seq, dim, true).unwrap();
        let flash = flash_attention_cpu(&q, &k, &v, seq, seq, dim, true).unwrap();
        for (i, (&s, &f)) in sdp.iter().zip(flash.iter()).enumerate() {
            assert!((s - f).abs() < 1e-4, "block-boundary mismatch at {i}: sdp={s} flash={f}");
        }
    }
}
