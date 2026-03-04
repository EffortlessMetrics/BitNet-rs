//! CPU quantized attention kernels with SIMD optimization.
//!
//! Provides INT8/INT4 quantized dot-product attention, multi-head
//! attention (MHA), grouped-query attention (GQA), KV-cache attention,
//! and an approximate flash-attention variant.  Each public function
//! performs runtime AVX2 detection and falls back to a scalar
//! implementation on platforms without AVX2.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

/// Construct an `InvalidArguments` kernel error.
fn invalid_arg(msg: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: msg.to_string() })
}

// ── Configuration ──────────────────────────────────────────────────

/// Quantization bit-width selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 8-bit symmetric quantization (scale = absmax / 127).
    Int8,
    /// 4-bit symmetric quantization (scale = absmax / 7).
    Int4,
}

/// Parameters for a quantized attention computation.
#[derive(Debug, Clone)]
pub struct QuantizedAttentionConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (for GQA; equals `num_heads` for MHA).
    pub num_kv_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Sequence length (tokens).
    pub seq_len: usize,
    /// Whether to apply a causal (upper-triangular) mask.
    pub causal: bool,
    /// Quantization bit-width.
    pub quant_bits: QuantBits,
    /// Optional explicit scale factor; defaults to `1 / sqrt(head_dim)`.
    pub scale: Option<f32>,
}

impl QuantizedAttentionConfig {
    /// Validate configuration, returning an error on inconsistencies.
    pub fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.num_kv_heads == 0 {
            return Err(invalid_arg("num_kv_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if self.seq_len == 0 {
            return Err(invalid_arg("seq_len must be > 0"));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(invalid_arg("num_heads must be a multiple of num_kv_heads"));
        }
        Ok(())
    }

    fn scale_factor(&self) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

/// Pre-quantized Q/K/V tensors with per-head scale factors.
#[derive(Debug, Clone)]
pub struct QuantizedQKV {
    /// Quantized query values (i8), shape `[num_heads * seq_len * head_dim]`.
    pub q_data: Vec<i8>,
    /// Per-head Q de-quantization scales, length `num_heads`.
    pub q_scales: Vec<f32>,
    /// Quantized key values (i8), shape `[num_kv_heads * seq_len * head_dim]`.
    pub k_data: Vec<i8>,
    /// Per-head K de-quantization scales, length `num_kv_heads`.
    pub k_scales: Vec<f32>,
    /// Quantized value values (i8), shape `[num_kv_heads * seq_len * head_dim]`.
    pub v_data: Vec<i8>,
    /// Per-head V de-quantization scales, length `num_kv_heads`.
    pub v_scales: Vec<f32>,
}

/// Pre-allocated workspace for quantized attention, avoiding
/// per-call `scores` allocation in the hot loop.
#[derive(Debug, Clone)]
pub struct QuantizedAttentionWorkspace {
    /// Scores buffer of at least `seq_len * seq_len` elements.
    pub scores: Vec<f32>,
}

impl QuantizedAttentionWorkspace {
    /// Create a workspace sized for the given sequence length.
    pub fn new(seq_len: usize) -> Self {
        Self { scores: vec![0.0f32; seq_len * seq_len] }
    }

    /// Ensure the workspace is large enough for the given sequence
    /// length, growing the buffer if needed.
    pub fn ensure_capacity(&mut self, seq_len: usize) {
        let required = seq_len * seq_len;
        if self.scores.len() < required {
            self.scores.resize(required, 0.0);
        }
    }
}

// ── SIMD: AVX2 i8 dot product ─────────────────────────────────────

/// AVX2-accelerated dot product of two `i8` slices.
///
/// Processes 32 elements per iteration using
/// `_mm256_cvtepi8_epi16` → `_mm256_madd_epi16` → horizontal sum.
///
/// # Safety
///
/// Caller must ensure `avx2` is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx2_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    let chunks = n / 32;
    let mut acc = _mm256_setzero_si256();

    for c in 0..chunks {
        let offset = c * 32;

        // Load 32 bytes from each input.
        let va = _mm256_loadu_si256(a.as_ptr().add(offset).cast());
        let vb = _mm256_loadu_si256(b.as_ptr().add(offset).cast());

        // Split into low / high 128-bit halves and sign-extend to i16.
        let a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(va));
        let b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        let b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vb));

        // Multiply-and-horizontal-add i16 pairs → i32.
        let prod_lo = _mm256_madd_epi16(a_lo, b_lo);
        let prod_hi = _mm256_madd_epi16(a_hi, b_hi);

        acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
    }

    // Horizontal sum of 8 × i32 lanes.
    let hi128 = _mm256_extracti128_si256::<1>(acc);
    let lo128 = _mm256_castsi256_si128(acc);
    let sum4 = _mm_add_epi32(hi128, lo128);
    let hi2 = _mm_unpackhi_epi64(sum4, sum4);
    let sum2 = _mm_add_epi32(sum4, hi2);
    let hi1 = _mm_shuffle_epi32::<0x01>(sum2);
    let mut result = _mm_cvtsi128_si32(_mm_add_epi32(sum2, hi1));

    // Scalar tail for remaining elements.
    for i in (chunks * 32)..n {
        result += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
    }
    result
}

/// Scalar fallback for i8 dot product.
fn scalar_dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Dispatch i8 dot product: AVX2 if available, else scalar.
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection ensures AVX2 is available.
            return unsafe { avx2_dot_i8(a, b) };
        }
    }
    scalar_dot_i8(a, b)
}

// ── Core attention helpers ─────────────────────────────────────────

/// Softmax in-place over a mutable slice.
fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Decode INT4 packed i8 value to the low nibble (sign-extended).
#[allow(dead_code)]
fn unpack_i4(val: i8) -> i8 {
    // Shift left 4, then arithmetic shift right 4 to sign-extend.
    (val << 4) >> 4
}

// ── Public API ─────────────────────────────────────────────────────

/// Compute quantized scaled dot-product attention for a single head,
/// writing intermediate scores into the caller-provided `scores_buf`.
///
/// `scores_buf` must have at least `seq_len * seq_len` elements.
/// `q`, `k`, `v` are quantized i8 matrices of shape `[seq_len, head_dim]`.
/// Returns `output` of shape `[seq_len, head_dim]` in f32.
#[allow(clippy::too_many_arguments)]
pub fn quantized_dot_product_attention_into(
    config: &QuantizedAttentionConfig,
    q: &[i8],
    k: &[i8],
    v: &[i8],
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    scores_buf: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    let s = config.seq_len;
    let d = config.head_dim;
    let expected = s * d;
    if q.len() < expected || k.len() < expected || v.len() < expected {
        return Err(invalid_arg("input tensor too small for seq_len * head_dim"));
    }
    if output.len() < expected {
        return Err(invalid_arg("output buffer too small"));
    }
    if scores_buf.len() < s * s {
        return Err(invalid_arg("scores buffer too small for seq_len * seq_len"));
    }

    let scale = config.scale_factor() * q_scale * k_scale;

    // Compute attention scores: scores[i][j] = scale * dot(Q[i], K[j]).
    let scores = &mut scores_buf[..s * s];
    for i in 0..s {
        for j in 0..s {
            if config.causal && j > i {
                scores[i * s + j] = f32::NEG_INFINITY;
            } else {
                let dot = dot_i8(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]);
                scores[i * s + j] = dot as f32 * scale;
            }
        }
        softmax_inplace(&mut scores[i * s..(i + 1) * s]);
    }

    // Weighted sum over V: output[i] = sum_j scores[i][j] * V[j].
    for i in 0..s {
        for dd in 0..d {
            let mut acc = 0.0f32;
            for j in 0..s {
                acc += scores[i * s + j] * v[j * d + dd] as f32 * v_scale;
            }
            output[i * d + dd] = acc;
        }
    }
    Ok(())
}

/// Compute quantized scaled dot-product attention for a single head.
///
/// `q`, `k`, `v` are quantized i8 matrices of shape `[seq_len, head_dim]`.
/// Returns `output` of shape `[seq_len, head_dim]` in f32.
pub fn quantized_dot_product_attention(
    config: &QuantizedAttentionConfig,
    q: &[i8],
    k: &[i8],
    v: &[i8],
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut [f32],
) -> Result<()> {
    let mut scores = vec![0.0f32; config.seq_len * config.seq_len];
    quantized_dot_product_attention_into(
        config,
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        &mut scores,
        output,
    )
}

/// Multi-head quantized attention.
///
/// Iterates over `num_heads` heads, dispatching
/// [`quantized_dot_product_attention`] per head.
pub fn quantized_multi_head_attention(
    config: &QuantizedAttentionConfig,
    qkv: &QuantizedQKV,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    if config.num_kv_heads != config.num_heads {
        return Err(invalid_arg("MHA requires num_kv_heads == num_heads; use GQA for grouped"));
    }
    let head_elems = config.seq_len * config.head_dim;
    let total = config.num_heads * head_elems;
    if qkv.q_data.len() < total || qkv.k_data.len() < total || qkv.v_data.len() < total {
        return Err(invalid_arg("QKV data too small for MHA dimensions"));
    }
    if output.len() < total {
        return Err(invalid_arg("output buffer too small for MHA"));
    }

    for h in 0..config.num_heads {
        let off = h * head_elems;
        quantized_dot_product_attention(
            config,
            &qkv.q_data[off..off + head_elems],
            &qkv.k_data[off..off + head_elems],
            &qkv.v_data[off..off + head_elems],
            qkv.q_scales[h],
            qkv.k_scales[h],
            qkv.v_scales[h],
            &mut output[off..off + head_elems],
        )?;
    }
    Ok(())
}

/// Grouped-query quantized attention.
///
/// Each group of `num_heads / num_kv_heads` query heads shares one
/// key/value head.
pub fn quantized_grouped_query_attention(
    config: &QuantizedAttentionConfig,
    qkv: &QuantizedQKV,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    let head_elems = config.seq_len * config.head_dim;
    let group_size = config.num_heads / config.num_kv_heads;
    let total_q = config.num_heads * head_elems;
    let total_kv = config.num_kv_heads * head_elems;

    if qkv.q_data.len() < total_q {
        return Err(invalid_arg("Q data too small for GQA"));
    }
    if qkv.k_data.len() < total_kv || qkv.v_data.len() < total_kv {
        return Err(invalid_arg("K/V data too small for GQA"));
    }
    if output.len() < total_q {
        return Err(invalid_arg("output too small for GQA"));
    }

    for h in 0..config.num_heads {
        let kv_h = h / group_size;
        let q_off = h * head_elems;
        let kv_off = kv_h * head_elems;
        quantized_dot_product_attention(
            config,
            &qkv.q_data[q_off..q_off + head_elems],
            &qkv.k_data[kv_off..kv_off + head_elems],
            &qkv.v_data[kv_off..kv_off + head_elems],
            qkv.q_scales[h],
            qkv.k_scales[kv_h],
            qkv.v_scales[kv_h],
            &mut output[q_off..q_off + head_elems],
        )?;
    }
    Ok(())
}

/// Multi-head quantized attention with a pre-allocated workspace.
///
/// Semantically identical to [`quantized_multi_head_attention`], but
/// reuses the scores buffer from `ws` instead of allocating per head.
pub fn quantized_multi_head_attention_with_workspace(
    config: &QuantizedAttentionConfig,
    qkv: &QuantizedQKV,
    ws: &mut QuantizedAttentionWorkspace,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    if config.num_kv_heads != config.num_heads {
        return Err(invalid_arg("MHA requires num_kv_heads == num_heads; use GQA for grouped"));
    }
    let head_elems = config.seq_len * config.head_dim;
    let total = config.num_heads * head_elems;
    if qkv.q_data.len() < total || qkv.k_data.len() < total || qkv.v_data.len() < total {
        return Err(invalid_arg("QKV data too small for MHA dimensions"));
    }
    if output.len() < total {
        return Err(invalid_arg("output buffer too small for MHA"));
    }

    ws.ensure_capacity(config.seq_len);

    for h in 0..config.num_heads {
        let off = h * head_elems;
        quantized_dot_product_attention_into(
            config,
            &qkv.q_data[off..off + head_elems],
            &qkv.k_data[off..off + head_elems],
            &qkv.v_data[off..off + head_elems],
            qkv.q_scales[h],
            qkv.k_scales[h],
            qkv.v_scales[h],
            &mut ws.scores,
            &mut output[off..off + head_elems],
        )?;
    }
    Ok(())
}

/// Grouped-query quantized attention with a pre-allocated workspace.
///
/// Semantically identical to [`quantized_grouped_query_attention`], but
/// reuses the scores buffer from `ws` instead of allocating per head.
pub fn quantized_grouped_query_attention_with_workspace(
    config: &QuantizedAttentionConfig,
    qkv: &QuantizedQKV,
    ws: &mut QuantizedAttentionWorkspace,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    let head_elems = config.seq_len * config.head_dim;
    let group_size = config.num_heads / config.num_kv_heads;
    let total_q = config.num_heads * head_elems;
    let total_kv = config.num_kv_heads * head_elems;

    if qkv.q_data.len() < total_q {
        return Err(invalid_arg("Q data too small for GQA"));
    }
    if qkv.k_data.len() < total_kv || qkv.v_data.len() < total_kv {
        return Err(invalid_arg("K/V data too small for GQA"));
    }
    if output.len() < total_q {
        return Err(invalid_arg("output too small for GQA"));
    }

    ws.ensure_capacity(config.seq_len);

    for h in 0..config.num_heads {
        let kv_h = h / group_size;
        let q_off = h * head_elems;
        let kv_off = kv_h * head_elems;
        quantized_dot_product_attention_into(
            config,
            &qkv.q_data[q_off..q_off + head_elems],
            &qkv.k_data[kv_off..kv_off + head_elems],
            &qkv.v_data[kv_off..kv_off + head_elems],
            qkv.q_scales[h],
            qkv.k_scales[kv_h],
            qkv.v_scales[kv_h],
            &mut ws.scores,
            &mut output[q_off..q_off + head_elems],
        )?;
    }
    Ok(())
}

/// De-quantize i8 Q/K/V and compute standard float attention.
///
/// Useful as a reference path: de-quantize first, then run f32
/// dot-product attention.
pub fn dequantize_and_attend(
    config: &QuantizedAttentionConfig,
    q_i8: &[i8],
    k_i8: &[i8],
    v_i8: &[i8],
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    let s = config.seq_len;
    let d = config.head_dim;
    let n = s * d;
    if q_i8.len() < n || k_i8.len() < n || v_i8.len() < n {
        return Err(invalid_arg("input too small for dequantize_and_attend"));
    }
    if output.len() < n {
        return Err(invalid_arg("output too small for dequantize_and_attend"));
    }

    let deq =
        |data: &[i8], scale: f32| -> Vec<f32> { data.iter().map(|&x| x as f32 * scale).collect() };
    let q_f = deq(&q_i8[..n], q_scale);
    let k_f = deq(&k_i8[..n], k_scale);
    let v_f = deq(&v_i8[..n], v_scale);

    let attn_scale = config.scale_factor();

    let mut scores = vec![0.0f32; s * s];
    for i in 0..s {
        for j in 0..s {
            if config.causal && j > i {
                scores[i * s + j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q_f[i * d + dd] * k_f[j * d + dd];
                }
                scores[i * s + j] = dot * attn_scale;
            }
        }
        softmax_inplace(&mut scores[i * s..(i + 1) * s]);
    }

    for i in 0..s {
        for dd in 0..d {
            let mut acc = 0.0f32;
            for j in 0..s {
                acc += scores[i * s + j] * v_f[j * d + dd];
            }
            output[i * d + dd] = acc;
        }
    }
    Ok(())
}

/// Compute raw quantized attention scores (before softmax).
///
/// Returns `scores` of shape `[seq_len, seq_len]`.
pub fn quantized_attention_scores(
    config: &QuantizedAttentionConfig,
    q: &[i8],
    k: &[i8],
    q_scale: f32,
    k_scale: f32,
    scores: &mut [f32],
) -> Result<()> {
    config.validate()?;
    let s = config.seq_len;
    let d = config.head_dim;
    if q.len() < s * d || k.len() < s * d {
        return Err(invalid_arg("Q/K too small for attention_scores"));
    }
    if scores.len() < s * s {
        return Err(invalid_arg("scores buffer too small"));
    }
    let scale = config.scale_factor() * q_scale * k_scale;
    for i in 0..s {
        for j in 0..s {
            if config.causal && j > i {
                scores[i * s + j] = f32::NEG_INFINITY;
            } else {
                let dot = dot_i8(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]);
                scores[i * s + j] = dot as f32 * scale;
            }
        }
    }
    Ok(())
}

/// Apply softmax to pre-computed scores, then weight V.
///
/// `scores` is `[seq_len, seq_len]`, `v` is `[seq_len, head_dim]`.
pub fn quantized_softmax_attention(
    config: &QuantizedAttentionConfig,
    scores: &mut [f32],
    v: &[i8],
    v_scale: f32,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    let s = config.seq_len;
    let d = config.head_dim;
    if scores.len() < s * s {
        return Err(invalid_arg("scores too small for softmax_attention"));
    }
    if v.len() < s * d {
        return Err(invalid_arg("V too small for softmax_attention"));
    }
    if output.len() < s * d {
        return Err(invalid_arg("output too small for softmax_attention"));
    }

    for i in 0..s {
        softmax_inplace(&mut scores[i * s..(i + 1) * s]);
    }

    for i in 0..s {
        for dd in 0..d {
            let mut acc = 0.0f32;
            for j in 0..s {
                acc += scores[i * s + j] * v[j * d + dd] as f32 * v_scale;
            }
            output[i * d + dd] = acc;
        }
    }
    Ok(())
}

/// Incremental KV-cache quantized attention.
///
/// `q_new` is a single new query row `[head_dim]`.
/// `k_cache` / `v_cache` are `[cache_len, head_dim]`.
/// Returns `output` of length `head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn quantized_kv_cache_attention(
    head_dim: usize,
    cache_len: usize,
    q_new: &[i8],
    k_cache: &[i8],
    v_cache: &[i8],
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    output: &mut [f32],
) -> Result<()> {
    if head_dim == 0 {
        return Err(invalid_arg("head_dim must be > 0"));
    }
    if cache_len == 0 {
        return Err(invalid_arg("cache_len must be > 0"));
    }
    if q_new.len() < head_dim {
        return Err(invalid_arg("q_new too small"));
    }
    if k_cache.len() < cache_len * head_dim {
        return Err(invalid_arg("k_cache too small"));
    }
    if v_cache.len() < cache_len * head_dim {
        return Err(invalid_arg("v_cache too small"));
    }
    if output.len() < head_dim {
        return Err(invalid_arg("output too small for kv_cache_attention"));
    }

    let scale = 1.0 / (head_dim as f32).sqrt() * q_scale * k_scale;

    let mut scores = vec![0.0f32; cache_len];
    for j in 0..cache_len {
        let dot = dot_i8(&q_new[..head_dim], &k_cache[j * head_dim..(j + 1) * head_dim]);
        scores[j] = dot as f32 * scale;
    }
    softmax_inplace(&mut scores);

    for dd in 0..head_dim {
        let mut acc = 0.0f32;
        for j in 0..cache_len {
            acc += scores[j] * v_cache[j * head_dim + dd] as f32 * v_scale;
        }
        output[dd] = acc;
    }
    Ok(())
}

/// Approximate flash-attention for quantized tensors.
///
/// Processes the sequence in blocks of `block_size` rows, maintaining
/// a running softmax numerically-stable accumulator.  Produces results
/// close to standard attention but with reduced peak memory.
#[allow(clippy::too_many_arguments)]
pub fn quantized_flash_attention_approx(
    config: &QuantizedAttentionConfig,
    q: &[i8],
    k: &[i8],
    v: &[i8],
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    block_size: usize,
    output: &mut [f32],
) -> Result<()> {
    config.validate()?;
    if block_size == 0 {
        return Err(invalid_arg("block_size must be > 0"));
    }
    let s = config.seq_len;
    let d = config.head_dim;
    let n = s * d;
    if q.len() < n || k.len() < n || v.len() < n {
        return Err(invalid_arg("input too small for flash_attention"));
    }
    if output.len() < n {
        return Err(invalid_arg("output too small for flash_attention"));
    }

    let scale = config.scale_factor() * q_scale * k_scale;

    for i in 0..s {
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; d];

        let j_end = if config.causal { i + 1 } else { s };

        for block_start in (0..j_end).step_by(block_size) {
            let block_end = (block_start + block_size).min(j_end);

            // Compute scores for this block.
            let blen = block_end - block_start;
            let mut block_scores = vec![0.0f32; blen];
            for (bj, j) in (block_start..block_end).enumerate() {
                let dot = dot_i8(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]);
                block_scores[bj] = dot as f32 * scale;
            }

            let block_max = block_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let new_max = running_max.max(block_max);

            // Rescale previous accumulator.
            if running_sum > 0.0 {
                let correction = (running_max - new_max).exp();
                running_sum *= correction;
                for a in acc.iter_mut() {
                    *a *= correction;
                }
            }

            // Accumulate this block.
            let mut block_sum = 0.0f32;
            for (bj, &score) in block_scores.iter().enumerate().take(blen) {
                let w = (score - new_max).exp();
                block_sum += w;
                let j = block_start + bj;
                for dd in 0..d {
                    acc[dd] += w * v[j * d + dd] as f32 * v_scale;
                }
            }

            running_max = new_max;
            running_sum += block_sum;
        }

        // Normalise.
        if running_sum > 0.0 {
            for dd in 0..d {
                output[i * d + dd] = acc[dd] / running_sum;
            }
        } else {
            output[i * d..i * d + d].fill(0.0);
        }
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(x, y, tol), "mismatch at index {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// Symmetric INT8 quantization helper for tests.
    fn quantize_i8(data: &[f32]) -> (Vec<i8>, f32) {
        let absmax = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        if absmax == 0.0 {
            return (vec![0i8; data.len()], 1.0);
        }
        let scale = absmax / 127.0;
        let quant: Vec<i8> =
            data.iter().map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8).collect();
        (quant, scale)
    }

    /// Symmetric INT4 quantization helper for tests.
    fn quantize_i4(data: &[f32]) -> (Vec<i8>, f32) {
        let absmax = data.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        if absmax == 0.0 {
            return (vec![0i8; data.len()], 1.0);
        }
        let scale = absmax / 7.0;
        let quant: Vec<i8> =
            data.iter().map(|&x| (x / scale).round().clamp(-7.0, 7.0) as i8).collect();
        (quant, scale)
    }

    /// Pure-float reference single-head attention.
    fn reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
            softmax_inplace(&mut scores[i * seq_len..(i + 1) * seq_len]);
        }
        let mut out = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[i * seq_len + j] * v[j * head_dim + d];
                }
                out[i * head_dim + d] = acc;
            }
        }
        out
    }

    fn make_config(
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        causal: bool,
    ) -> QuantizedAttentionConfig {
        QuantizedAttentionConfig {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            seq_len,
            causal,
            quant_bits: QuantBits::Int8,
            scale: None,
        }
    }

    fn make_gqa_config(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> QuantizedAttentionConfig {
        QuantizedAttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
            causal: false,
            quant_bits: QuantBits::Int8,
            scale: None,
        }
    }

    // ── Config validation ──────────────────────────────────────────

    #[test]
    fn config_valid_basic() {
        make_config(4, 64, 8, false).validate().unwrap();
    }

    #[test]
    fn config_zero_heads() {
        assert!(make_config(0, 64, 8, false).validate().is_err());
    }

    #[test]
    fn config_zero_head_dim() {
        assert!(make_config(4, 0, 8, false).validate().is_err());
    }

    #[test]
    fn config_zero_seq_len() {
        assert!(make_config(4, 64, 0, false).validate().is_err());
    }

    #[test]
    fn config_zero_kv_heads() {
        let mut c = make_config(4, 64, 8, false);
        c.num_kv_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_heads_not_multiple_of_kv() {
        let mut c = make_config(5, 64, 8, false);
        c.num_kv_heads = 3;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_gqa_valid() {
        make_gqa_config(8, 2, 64, 4).validate().unwrap();
    }

    #[test]
    fn config_scale_factor_default() {
        let c = make_config(1, 64, 1, false);
        let s = c.scale_factor();
        assert!(approx_eq(s, 1.0 / 8.0, 1e-6));
    }

    #[test]
    fn config_scale_factor_custom() {
        let mut c = make_config(1, 64, 1, false);
        c.scale = Some(0.5);
        assert!(approx_eq(c.scale_factor(), 0.5, 1e-6));
    }

    // ── Scalar dot i8 ──────────────────────────────────────────────

    #[test]
    fn scalar_dot_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        assert_eq!(scalar_dot_i8(&a, &b), 70);
    }

    #[test]
    fn scalar_dot_negative() {
        let a: Vec<i8> = vec![-1, 2, -3];
        let b: Vec<i8> = vec![4, -5, 6];
        // -4 + -10 + -18 = -32
        assert_eq!(scalar_dot_i8(&a, &b), -32);
    }

    #[test]
    fn scalar_dot_empty() {
        assert_eq!(scalar_dot_i8(&[], &[]), 0);
    }

    #[test]
    fn scalar_dot_single() {
        assert_eq!(scalar_dot_i8(&[127], &[127]), 127 * 127);
    }

    // ── dot_i8 dispatch ────────────────────────────────────────────

    #[test]
    fn dot_i8_small() {
        let a: Vec<i8> = (1..=16).collect();
        let b: Vec<i8> = (1..=16).collect();
        let expected: i32 = (1..=16).map(|x: i32| x * x).sum();
        assert_eq!(dot_i8(&a, &b), expected);
    }

    #[test]
    fn dot_i8_large_aligned() {
        // 64 elements — exercises AVX2 path (2 chunks of 32).
        let a: Vec<i8> = (0..64).map(|i| ((i % 11) as i8) - 5).collect();
        let b: Vec<i8> = (0..64).map(|i| ((i % 7) as i8) - 3).collect();
        let expected = scalar_dot_i8(&a, &b);
        assert_eq!(dot_i8(&a, &b), expected);
    }

    #[test]
    fn dot_i8_unaligned_tail() {
        // 50 elements — 1 chunk of 32 + 18 scalar tail.
        let a: Vec<i8> = (0..50).map(|i| (i as i8) - 25).collect();
        let b: Vec<i8> = (0..50).map(|i| ((i * 3) as i8) % 10).collect();
        let expected = scalar_dot_i8(&a, &b);
        assert_eq!(dot_i8(&a, &b), expected);
    }

    #[test]
    fn dot_i8_zeros() {
        let a = vec![0i8; 64];
        let b = vec![127i8; 64];
        assert_eq!(dot_i8(&a, &b), 0);
    }

    #[test]
    fn dot_i8_extremes() {
        let a = vec![127i8; 32];
        let b = vec![-128i8; 32];
        let expected: i32 = 32 * (127 * -128);
        assert_eq!(dot_i8(&a, &b), expected);
    }

    // ── softmax ────────────────────────────────────────────────────

    #[test]
    fn softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS));
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn softmax_single() {
        let mut v = vec![42.0];
        softmax_inplace(&mut v);
        assert!(approx_eq(v[0], 1.0, EPS));
    }

    #[test]
    fn softmax_uniform() {
        let mut v = vec![5.0; 4];
        softmax_inplace(&mut v);
        for &x in &v {
            assert!(approx_eq(x, 0.25, EPS));
        }
    }

    #[test]
    fn softmax_large_values() {
        let mut v = vec![1000.0, 1001.0, 1002.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS));
    }

    // ── unpack_i4 ──────────────────────────────────────────────────

    #[test]
    fn unpack_i4_positive() {
        assert_eq!(unpack_i4(5), 5);
    }

    #[test]
    fn unpack_i4_negative() {
        // -3 in low nibble: 0xFD -> low nibble is 0xD = -3 sign-extended
        assert_eq!(unpack_i4(-3), -3);
    }

    #[test]
    fn unpack_i4_zero() {
        assert_eq!(unpack_i4(0), 0);
    }

    #[test]
    fn unpack_i4_max() {
        assert_eq!(unpack_i4(7), 7);
    }

    #[test]
    fn unpack_i4_min() {
        assert_eq!(unpack_i4(-8), -8);
    }

    // ── quantized_dot_product_attention ─────────────────────────────

    #[test]
    fn qda_identity_scale() {
        let cfg = make_config(1, 4, 2, false);
        let q: Vec<i8> = vec![10, 0, 0, 0, 0, 10, 0, 0];
        let k: Vec<i8> = vec![10, 0, 0, 0, 0, 10, 0, 0];
        let v: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut out = vec![0.0f32; 8];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        // Output should be valid f32 with reasonable values.
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn qda_causal_mask() {
        let cfg = make_config(1, 4, 3, true);
        let q = vec![1i8; 12];
        let k = vec![1i8; 12];
        let v = vec![1i8; 12];
        let mut out = vec![0.0f32; 12];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn qda_input_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let q = vec![0i8; 4]; // need 8
        let k = vec![0i8; 8];
        let v = vec![0i8; 8];
        let mut out = vec![0.0f32; 8];
        assert!(
            quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out,).is_err()
        );
    }

    #[test]
    fn qda_output_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let q = vec![0i8; 8];
        let k = vec![0i8; 8];
        let v = vec![0i8; 8];
        let mut out = vec![0.0f32; 4]; // need 8
        assert!(
            quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out,).is_err()
        );
    }

    // ── quantized_multi_head_attention ──────────────────────────────

    #[test]
    fn mha_two_heads() {
        let cfg = make_config(2, 4, 2, false);
        let he = 2 * 4; // seq_len * head_dim
        let total = 2 * he;
        let qkv = QuantizedQKV {
            q_data: vec![1i8; total],
            q_scales: vec![1.0; 2],
            k_data: vec![1i8; total],
            k_scales: vec![1.0; 2],
            v_data: vec![1i8; total],
            v_scales: vec![1.0; 2],
        };
        let mut out = vec![0.0f32; total];
        quantized_multi_head_attention(&cfg, &qkv, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mha_rejects_gqa_config() {
        let mut cfg = make_config(4, 4, 2, false);
        cfg.num_kv_heads = 2;
        let he = 2 * 4;
        let qkv = QuantizedQKV {
            q_data: vec![0i8; 4 * he],
            q_scales: vec![1.0; 4],
            k_data: vec![0i8; 2 * he],
            k_scales: vec![1.0; 2],
            v_data: vec![0i8; 2 * he],
            v_scales: vec![1.0; 2],
        };
        let mut out = vec![0.0f32; 4 * he];
        assert!(quantized_multi_head_attention(&cfg, &qkv, &mut out).is_err());
    }

    #[test]
    fn mha_data_too_small() {
        let cfg = make_config(2, 4, 2, false);
        let qkv = QuantizedQKV {
            q_data: vec![0i8; 4], // too small
            q_scales: vec![1.0; 2],
            k_data: vec![0i8; 16],
            k_scales: vec![1.0; 2],
            v_data: vec![0i8; 16],
            v_scales: vec![1.0; 2],
        };
        let mut out = vec![0.0f32; 16];
        assert!(quantized_multi_head_attention(&cfg, &qkv, &mut out).is_err());
    }

    // ── quantized_grouped_query_attention ───────────────────────────

    #[test]
    fn gqa_4_heads_2_kv() {
        let cfg = make_gqa_config(4, 2, 4, 2);
        let he = 2 * 4;
        let qkv = QuantizedQKV {
            q_data: vec![1i8; 4 * he],
            q_scales: vec![1.0; 4],
            k_data: vec![1i8; 2 * he],
            k_scales: vec![1.0; 2],
            v_data: vec![1i8; 2 * he],
            v_scales: vec![1.0; 2],
        };
        let mut out = vec![0.0f32; 4 * he];
        quantized_grouped_query_attention(&cfg, &qkv, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gqa_shared_heads_match() {
        // With 4 heads and 2 kv heads, heads 0&1 share kv[0], 2&3 share kv[1].
        let cfg = make_gqa_config(4, 2, 4, 1);
        let he = 1 * 4;
        let qkv = QuantizedQKV {
            q_data: vec![1i8; 4 * he],
            q_scales: vec![1.0; 4],
            k_data: vec![1i8; 2 * he],
            k_scales: vec![1.0; 2],
            v_data: vec![1i8; 2 * he],
            v_scales: vec![1.0; 2],
        };
        let mut out = vec![0.0f32; 4 * he];
        quantized_grouped_query_attention(&cfg, &qkv, &mut out).unwrap();
        // Heads 0 and 1 share KV, so their outputs should match.
        assert_close(&out[0..he], &out[he..2 * he], EPS);
        // Heads 2 and 3 share KV, so they should match too.
        assert_close(&out[2 * he..3 * he], &out[3 * he..4 * he], EPS);
    }

    #[test]
    fn gqa_q_too_small() {
        let cfg = make_gqa_config(4, 2, 4, 2);
        let he = 2 * 4;
        let qkv = QuantizedQKV {
            q_data: vec![0i8; 2 * he], // need 4 * he
            q_scales: vec![1.0; 4],
            k_data: vec![0i8; 2 * he],
            k_scales: vec![1.0; 2],
            v_data: vec![0i8; 2 * he],
            v_scales: vec![1.0; 2],
        };
        let mut out = vec![0.0f32; 4 * he];
        assert!(quantized_grouped_query_attention(&cfg, &qkv, &mut out).is_err());
    }

    // ── dequantize_and_attend ──────────────────────────────────────

    #[test]
    fn deq_attend_matches_float_ref() {
        let seq = 3;
        let dim = 8;
        // Random-ish float data.
        let q_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32 * 0.1) - 1.2).collect();
        let k_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32 * 0.07) + 0.3).collect();
        let v_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32 * 0.05) - 0.5).collect();

        let (q_q, q_s) = quantize_i8(&q_f);
        let (k_q, k_s) = quantize_i8(&k_f);
        let (v_q, v_s) = quantize_i8(&v_f);

        let cfg = make_config(1, dim, seq, false);
        let mut out = vec![0.0f32; seq * dim];
        dequantize_and_attend(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, &mut out).unwrap();

        let ref_out = reference_attention(&q_f, &k_f, &v_f, seq, dim, false);
        // Quantization introduces error; allow 0.15 tolerance.
        assert_close(&out, &ref_out, 0.15);
    }

    #[test]
    fn deq_attend_causal() {
        let seq = 4;
        let dim = 4;
        let q_f: Vec<f32> = (0..seq * dim).map(|i| i as f32 * 0.1).collect();
        let k_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.05).collect();
        let v_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.02).collect();

        let (q_q, q_s) = quantize_i8(&q_f);
        let (k_q, k_s) = quantize_i8(&k_f);
        let (v_q, v_s) = quantize_i8(&v_f);

        let cfg = make_config(1, dim, seq, true);
        let mut out = vec![0.0f32; seq * dim];
        dequantize_and_attend(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // ── quantized_attention_scores ─────────────────────────────────

    #[test]
    fn scores_shape_and_causal() {
        let seq = 3;
        let dim = 4;
        let cfg = make_config(1, dim, seq, true);
        let q = vec![1i8; seq * dim];
        let k = vec![1i8; seq * dim];
        let mut scores = vec![0.0f32; seq * seq];
        quantized_attention_scores(&cfg, &q, &k, 1.0, 1.0, &mut scores).unwrap();
        // Upper triangle should be -inf.
        assert_eq!(scores[0 * seq + 1], f32::NEG_INFINITY);
        assert_eq!(scores[0 * seq + 2], f32::NEG_INFINITY);
        assert_eq!(scores[1 * seq + 2], f32::NEG_INFINITY);
        // Diagonal and below should be finite.
        assert!(scores[0 * seq + 0].is_finite());
        assert!(scores[1 * seq + 0].is_finite());
        assert!(scores[1 * seq + 1].is_finite());
    }

    #[test]
    fn scores_non_causal_all_finite() {
        let seq = 4;
        let dim = 4;
        let cfg = make_config(1, dim, seq, false);
        let q = vec![2i8; seq * dim];
        let k = vec![3i8; seq * dim];
        let mut scores = vec![0.0f32; seq * seq];
        quantized_attention_scores(&cfg, &q, &k, 1.0, 1.0, &mut scores).unwrap();
        assert!(scores.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn scores_buffer_too_small() {
        let cfg = make_config(1, 4, 3, false);
        let q = vec![0i8; 12];
        let k = vec![0i8; 12];
        let mut scores = vec![0.0f32; 4]; // need 9
        assert!(quantized_attention_scores(&cfg, &q, &k, 1.0, 1.0, &mut scores).is_err());
    }

    // ── quantized_softmax_attention ────────────────────────────────

    #[test]
    fn softmax_attn_basic() {
        let seq = 2;
        let dim = 4;
        let cfg = make_config(1, dim, seq, false);
        let mut scores = vec![1.0f32; seq * seq];
        let v = vec![10i8; seq * dim];
        let mut out = vec![0.0f32; seq * dim];
        quantized_softmax_attention(&cfg, &mut scores, &v, 1.0, &mut out).unwrap();
        // With uniform scores and uniform V, output should be V * v_scale.
        for &x in &out {
            assert!(approx_eq(x, 10.0, EPS));
        }
    }

    #[test]
    fn softmax_attn_v_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let mut scores = vec![0.0f32; 4];
        let v = vec![0i8; 2]; // too small
        let mut out = vec![0.0f32; 8];
        assert!(quantized_softmax_attention(&cfg, &mut scores, &v, 1.0, &mut out).is_err());
    }

    // ── quantized_kv_cache_attention ───────────────────────────────

    #[test]
    fn kv_cache_basic() {
        let dim = 4;
        let cache_len = 3;
        let q = vec![1i8; dim];
        let k = vec![1i8; cache_len * dim];
        let v = vec![2i8; cache_len * dim];
        let mut out = vec![0.0f32; dim];
        quantized_kv_cache_attention(dim, cache_len, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        // Uniform Q/K → uniform scores → output ≈ mean(V) * v_scale = 2.
        for &x in &out {
            assert!(approx_eq(x, 2.0, EPS));
        }
    }

    #[test]
    fn kv_cache_single_entry() {
        let dim = 4;
        let q = vec![10i8; dim];
        let k = vec![10i8; dim];
        let v: Vec<i8> = vec![1, 2, 3, 4];
        let mut out = vec![0.0f32; dim];
        quantized_kv_cache_attention(dim, 1, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        // Single cache entry → softmax([score]) = [1.0] → output = V.
        let expected: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        assert_close(&out, &expected, EPS);
    }

    #[test]
    fn kv_cache_zero_dim() {
        let mut out = vec![0.0f32; 0];
        assert!(
            quantized_kv_cache_attention(0, 1, &[], &[], &[], 1.0, 1.0, 1.0, &mut out,).is_err()
        );
    }

    #[test]
    fn kv_cache_zero_cache() {
        let mut out = vec![0.0f32; 4];
        assert!(
            quantized_kv_cache_attention(4, 0, &[1; 4], &[], &[], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }

    #[test]
    fn kv_cache_q_too_small() {
        let mut out = vec![0.0f32; 4];
        assert!(
            quantized_kv_cache_attention(4, 2, &[1; 2], &[1; 8], &[1; 8], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }

    // ── quantized_flash_attention_approx ───────────────────────────

    #[test]
    fn flash_matches_standard() {
        let seq = 4;
        let dim = 8;
        let q_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let k_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.08 + 0.2).collect();
        let v_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.06 - 0.3).collect();

        let (q_q, q_s) = quantize_i8(&q_f);
        let (k_q, k_s) = quantize_i8(&k_f);
        let (v_q, v_s) = quantize_i8(&v_f);

        let cfg = make_config(1, dim, seq, false);

        let mut standard = vec![0.0f32; seq * dim];
        quantized_dot_product_attention(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, &mut standard)
            .unwrap();

        let mut flash = vec![0.0f32; seq * dim];
        quantized_flash_attention_approx(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, 2, &mut flash)
            .unwrap();

        assert_close(&flash, &standard, EPS);
    }

    #[test]
    fn flash_causal_matches_standard() {
        let seq = 4;
        let dim = 4;
        let q_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.12 - 0.8).collect();
        let k_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.09).collect();
        let v_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.04 + 0.1).collect();

        let (q_q, q_s) = quantize_i8(&q_f);
        let (k_q, k_s) = quantize_i8(&k_f);
        let (v_q, v_s) = quantize_i8(&v_f);

        let cfg = make_config(1, dim, seq, true);

        let mut standard = vec![0.0f32; seq * dim];
        quantized_dot_product_attention(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, &mut standard)
            .unwrap();

        let mut flash = vec![0.0f32; seq * dim];
        quantized_flash_attention_approx(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, 2, &mut flash)
            .unwrap();

        assert_close(&flash, &standard, EPS);
    }

    #[test]
    fn flash_block_size_one() {
        let seq = 3;
        let dim = 4;
        let cfg = make_config(1, dim, seq, false);
        let q = vec![1i8; seq * dim];
        let k = vec![1i8; seq * dim];
        let v = vec![2i8; seq * dim];
        let mut out = vec![0.0f32; seq * dim];
        quantized_flash_attention_approx(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, 1, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn flash_block_size_larger_than_seq() {
        let seq = 2;
        let dim = 4;
        let cfg = make_config(1, dim, seq, false);
        let q = vec![1i8; seq * dim];
        let k = vec![1i8; seq * dim];
        let v = vec![3i8; seq * dim];
        let mut out = vec![0.0f32; seq * dim];
        quantized_flash_attention_approx(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, 100, &mut out).unwrap();
        // Should behave identically to standard attention.
        let mut standard = vec![0.0f32; seq * dim];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut standard).unwrap();
        assert_close(&out, &standard, EPS);
    }

    #[test]
    fn flash_zero_block_size() {
        let cfg = make_config(1, 4, 2, false);
        let d = vec![0i8; 8];
        let mut out = vec![0.0f32; 8];
        assert!(
            quantized_flash_attention_approx(&cfg, &d, &d, &d, 1.0, 1.0, 1.0, 0, &mut out,)
                .is_err()
        );
    }

    // ── INT8 vs float reference accuracy ───────────────────────────

    #[test]
    fn int8_accuracy_vs_float() {
        let seq = 4;
        let dim = 16;
        let q_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let k_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.02 + 0.5).collect();
        let v_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.01 - 0.2).collect();

        let (q_q, q_s) = quantize_i8(&q_f);
        let (k_q, k_s) = quantize_i8(&k_f);
        let (v_q, v_s) = quantize_i8(&v_f);

        let cfg = make_config(1, dim, seq, false);
        let mut out = vec![0.0f32; seq * dim];
        quantized_dot_product_attention(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, &mut out).unwrap();

        let ref_out = reference_attention(&q_f, &k_f, &v_f, seq, dim, false);

        let max_err =
            out.iter().zip(ref_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 0.15, "INT8 max error {max_err} exceeds 0.15 threshold");
    }

    // ── INT4 accuracy ──────────────────────────────────────────────

    #[test]
    fn int4_accuracy_vs_float() {
        let seq = 3;
        let dim = 8;
        let q_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let k_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.08).collect();
        let v_f: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.05 + 0.1).collect();

        let (q_q, q_s) = quantize_i4(&q_f);
        let (k_q, k_s) = quantize_i4(&k_f);
        let (v_q, v_s) = quantize_i4(&v_f);

        let cfg = QuantizedAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: dim,
            seq_len: seq,
            causal: false,
            quant_bits: QuantBits::Int4,
            scale: None,
        };
        let mut out = vec![0.0f32; seq * dim];
        quantized_dot_product_attention(&cfg, &q_q, &k_q, &v_q, q_s, k_s, v_s, &mut out).unwrap();

        let ref_out = reference_attention(&q_f, &k_f, &v_f, seq, dim, false);
        let max_err =
            out.iter().zip(ref_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 0.35, "INT4 max error {max_err} exceeds 0.35 threshold");
    }

    // ── QuantBits enum ─────────────────────────────────────────────

    #[test]
    fn quant_bits_equality() {
        assert_eq!(QuantBits::Int8, QuantBits::Int8);
        assert_ne!(QuantBits::Int8, QuantBits::Int4);
    }

    #[test]
    fn quant_bits_debug() {
        let s = format!("{:?}", QuantBits::Int4);
        assert!(s.contains("Int4"));
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn single_token_single_dim() {
        let cfg = make_config(1, 1, 1, false);
        let q = vec![100i8];
        let k = vec![50i8];
        let v = vec![10i8];
        let mut out = vec![0.0f32; 1];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        // Single token: softmax of one element = 1.0 → out = V * v_scale.
        assert!(approx_eq(out[0], 10.0, EPS));
    }

    #[test]
    fn all_zeros_input() {
        let cfg = make_config(1, 4, 2, false);
        let q = vec![0i8; 8];
        let k = vec![0i8; 8];
        let v = vec![0i8; 8];
        let mut out = vec![0.0f32; 8];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        // All-zero Q/K → all scores 0 → uniform softmax → mean(V)=0.
        for &x in &out {
            assert!(approx_eq(x, 0.0, EPS));
        }
    }

    #[test]
    fn scale_zero_does_not_panic() {
        let cfg = QuantizedAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            seq_len: 2,
            causal: false,
            quant_bits: QuantBits::Int8,
            scale: Some(0.0),
        };
        let q = vec![1i8; 8];
        let k = vec![1i8; 8];
        let v = vec![1i8; 8];
        let mut out = vec![0.0f32; 8];
        // scale=0 → all scores 0 → uniform softmax → should not panic.
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn large_head_dim_avx2_coverage() {
        // 256 elements forces multiple AVX2 chunks.
        let dim = 256;
        let cfg = make_config(1, dim, 2, false);
        let q: Vec<i8> = (0..2 * dim).map(|i| ((i % 13) as i8) - 6).collect();
        let k: Vec<i8> = (0..2 * dim).map(|i| ((i % 11) as i8) - 5).collect();
        let v: Vec<i8> = (0..2 * dim).map(|i| ((i % 9) as i8) - 4).collect();
        let mut out = vec![0.0f32; 2 * dim];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 0.01, 0.01, 0.01, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn kv_cache_large_cache() {
        let dim = 32;
        let cache_len = 100;
        let q: Vec<i8> = (0..dim).map(|i| ((i % 7) as i8) - 3).collect();
        let k: Vec<i8> = (0..cache_len * dim).map(|i| ((i % 5) as i8) - 2).collect();
        let v: Vec<i8> = (0..cache_len * dim).map(|i| ((i % 9) as i8) - 4).collect();
        let mut out = vec![0.0f32; dim];
        quantized_kv_cache_attention(dim, cache_len, &q, &k, &v, 0.1, 0.1, 0.1, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mha_four_heads_causal() {
        let cfg = make_config(4, 8, 3, true);
        let he = 3 * 8;
        let total = 4 * he;
        let qkv = QuantizedQKV {
            q_data: (0..total).map(|i| ((i % 11) as i8) - 5).collect(),
            q_scales: vec![0.05; 4],
            k_data: (0..total).map(|i| ((i % 7) as i8) - 3).collect(),
            k_scales: vec![0.05; 4],
            v_data: (0..total).map(|i| ((i % 9) as i8) - 4).collect(),
            v_scales: vec![0.05; 4],
        };
        let mut out = vec![0.0f32; total];
        quantized_multi_head_attention(&cfg, &qkv, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn deq_attend_input_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let mut out = vec![0.0f32; 8];
        assert!(
            dequantize_and_attend(&cfg, &[0i8; 4], &[0i8; 8], &[0i8; 8], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }

    #[test]
    fn flash_attn_input_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let mut out = vec![0.0f32; 8];
        assert!(
            quantized_flash_attention_approx(
                &cfg, &[0i8; 4], &[0i8; 8], &[0i8; 8], 1.0, 1.0, 1.0, 2, &mut out,
            )
            .is_err()
        );
    }

    // ── Additional coverage ────────────────────────────────────────

    #[test]
    fn qda_non_causal_symmetric_qk() {
        let cfg = make_config(1, 4, 2, false);
        let q: Vec<i8> = vec![100, 0, 0, 0, 0, 100, 0, 0];
        let k: Vec<i8> = vec![100, 0, 0, 0, 0, 100, 0, 0];
        let v: Vec<i8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let mut out = vec![0.0f32; 8];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 0.01, 0.01, 1.0, &mut out).unwrap();
        assert!(out[0] < out[4], "row 0 should lean toward V[0]");
    }

    #[test]
    fn qda_large_seq_len() {
        let seq = 32;
        let dim = 8;
        let cfg = make_config(1, dim, seq, false);
        let n = seq * dim;
        let q: Vec<i8> = (0..n).map(|i| ((i % 13) as i8) - 6).collect();
        let k: Vec<i8> = (0..n).map(|i| ((i % 7) as i8) - 3).collect();
        let v: Vec<i8> = (0..n).map(|i| ((i % 9) as i8) - 4).collect();
        let mut out = vec![0.0f32; n];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 0.01, 0.01, 0.1, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn qda_negative_scale() {
        let cfg = QuantizedAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            seq_len: 2,
            causal: false,
            quant_bits: QuantBits::Int8,
            scale: Some(-1.0),
        };
        let q = vec![1i8; 8];
        let k = vec![1i8; 8];
        let v = vec![1i8; 8];
        let mut out = vec![0.0f32; 8];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut out).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mha_single_head_matches_dot_product() {
        let cfg = make_config(1, 4, 2, false);
        let q = vec![3i8; 8];
        let k = vec![2i8; 8];
        let v = vec![5i8; 8];
        let qkv = QuantizedQKV {
            q_data: q.clone(),
            q_scales: vec![1.0],
            k_data: k.clone(),
            k_scales: vec![1.0],
            v_data: v.clone(),
            v_scales: vec![1.0],
        };
        let mut mha_out = vec![0.0f32; 8];
        quantized_multi_head_attention(&cfg, &qkv, &mut mha_out).unwrap();

        let mut dp_out = vec![0.0f32; 8];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut dp_out).unwrap();
        assert_close(&mha_out, &dp_out, EPS);
    }

    #[test]
    fn gqa_single_kv_head() {
        let cfg = make_gqa_config(4, 1, 4, 2);
        let he = 2 * 4;
        let qkv = QuantizedQKV {
            q_data: vec![1i8; 4 * he],
            q_scales: vec![1.0; 4],
            k_data: vec![1i8; he],
            k_scales: vec![1.0; 1],
            v_data: vec![1i8; he],
            v_scales: vec![1.0; 1],
        };
        let mut out = vec![0.0f32; 4 * he];
        quantized_grouped_query_attention(&cfg, &qkv, &mut out).unwrap();
        for h in 1..4 {
            assert_close(&out[0..he], &out[h * he..(h + 1) * he], EPS);
        }
    }

    #[test]
    fn scores_scale_applied() {
        let cfg = QuantizedAttentionConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            seq_len: 1,
            causal: false,
            quant_bits: QuantBits::Int8,
            scale: Some(2.0),
        };
        let q = vec![1i8; 4];
        let k = vec![1i8; 4];
        let mut scores = vec![0.0f32; 1];
        quantized_attention_scores(&cfg, &q, &k, 1.0, 1.0, &mut scores).unwrap();
        assert!(approx_eq(scores[0], 8.0, EPS));
    }

    #[test]
    fn softmax_attn_weighted() {
        let seq = 2;
        let dim = 2;
        let cfg = make_config(1, dim, seq, false);
        let mut scores = vec![0.0, 100.0, 0.0, 100.0];
        let v: Vec<i8> = vec![0, 0, 10, 20];
        let mut out = vec![0.0f32; seq * dim];
        quantized_softmax_attention(&cfg, &mut scores, &v, 1.0, &mut out).unwrap();
        assert!(approx_eq(out[0], 10.0, 0.01));
        assert!(approx_eq(out[1], 20.0, 0.01));
    }

    #[test]
    fn kv_cache_with_scale() {
        let dim = 4;
        let q = vec![10i8; dim];
        let k = vec![10i8; dim];
        let v: Vec<i8> = vec![1, 2, 3, 4];
        let mut out = vec![0.0f32; dim];
        quantized_kv_cache_attention(dim, 1, &q, &k, &v, 0.5, 0.5, 2.0, &mut out).unwrap();
        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert_close(&out, &expected, EPS);
    }

    #[test]
    fn kv_cache_output_too_small() {
        let mut out = vec![0.0f32; 2];
        assert!(
            quantized_kv_cache_attention(4, 1, &[1; 4], &[1; 4], &[1; 4], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }

    #[test]
    fn flash_multiple_blocks() {
        let seq = 8;
        let dim = 4;
        let cfg = make_config(1, dim, seq, false);
        let n = seq * dim;
        let q: Vec<i8> = (0..n).map(|i| ((i % 5) as i8) - 2).collect();
        let k: Vec<i8> = (0..n).map(|i| ((i % 7) as i8) - 3).collect();
        let v: Vec<i8> = (0..n).map(|i| ((i % 3) as i8) - 1).collect();

        let mut standard = vec![0.0f32; n];
        quantized_dot_product_attention(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, &mut standard).unwrap();

        let mut flash = vec![0.0f32; n];
        quantized_flash_attention_approx(&cfg, &q, &k, &v, 1.0, 1.0, 1.0, 3, &mut flash).unwrap();
        assert_close(&flash, &standard, EPS);
    }

    #[test]
    fn deq_attend_output_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let mut out = vec![0.0f32; 4];
        assert!(
            dequantize_and_attend(&cfg, &[0i8; 8], &[0i8; 8], &[0i8; 8], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }

    #[test]
    fn flash_output_too_small() {
        let cfg = make_config(1, 4, 2, false);
        let mut out = vec![0.0f32; 4];
        assert!(
            quantized_flash_attention_approx(
                &cfg, &[0i8; 8], &[0i8; 8], &[0i8; 8], 1.0, 1.0, 1.0, 2, &mut out,
            )
            .is_err()
        );
    }

    #[test]
    fn dot_i8_length_mismatch_uses_min() {
        let a = vec![1i8; 10];
        let b = vec![2i8; 5];
        assert_eq!(dot_i8(&a, &b), 10);
    }

    #[test]
    fn config_equal_heads_and_kv() {
        make_gqa_config(8, 8, 32, 4).validate().unwrap();
    }

    #[test]
    fn softmax_with_neg_infinity() {
        let mut v = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        softmax_inplace(&mut v);
        assert!(approx_eq(v[0], 0.0, EPS));
        assert!(approx_eq(v[1], 1.0, EPS));
        assert!(approx_eq(v[2], 0.0, EPS));
    }

    #[test]
    fn kv_cache_k_too_small() {
        let mut out = vec![0.0f32; 4];
        assert!(
            quantized_kv_cache_attention(4, 2, &[1; 4], &[1; 4], &[1; 8], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }

    #[test]
    fn kv_cache_v_too_small() {
        let mut out = vec![0.0f32; 4];
        assert!(
            quantized_kv_cache_attention(4, 2, &[1; 4], &[1; 8], &[1; 4], 1.0, 1.0, 1.0, &mut out,)
                .is_err()
        );
    }
}
