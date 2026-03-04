//! CPU SIMD quantized attention kernels.
//!
//! Provides int8/int4 self-attention with AVX2 acceleration and scalar
//! fallback.  All public entry points auto-dispatch to the fastest
//! available code path at runtime via
//! `std::arch::is_x86_feature_detected!("avx2")` on x86_64 or fall back
//! to portable scalar loops on other architectures.
//!
//! # Pipeline
//!
//! 1. **quantized_qkv_projection** – project input to Q, K, V in int8
//! 2. **quantized_score_computation** – integer dot-product attention scores
//! 3. **quantized_softmax** – fixed-point softmax approximation
//! 4. **quantized_value_aggregation** – weighted sum in quantized domain
//! 5. **dequantized_output** – convert back to f32
//! 6. **quantized_self_attention** – full pipeline combining all stages
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

// ── Configuration ──────────────────────────────────────────────────────

/// Parameters for a quantized self-attention invocation.
#[derive(Debug, Clone)]
pub struct QuantizedAttentionConfig {
    /// Sequence length (number of tokens).
    pub seq_len: usize,
    /// Model / hidden dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Scale factor applied to scores (typically `1/sqrt(head_dim)`).
    /// When `None`, auto-computed as `1/sqrt(head_dim)`.
    pub scale: Option<f32>,
    /// Whether to apply a causal (lower-triangular) mask.
    pub causal: bool,
}

impl QuantizedAttentionConfig {
    /// Create a config with sensible defaults (no causal mask, auto-scale).
    pub fn new(seq_len: usize, head_dim: usize, num_heads: usize) -> Self {
        Self { seq_len, head_dim, num_heads, scale: None, causal: false }
    }

    fn effective_scale(&self) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

// ── Quantization helpers ───────────────────────────────────────────────

/// Result of symmetric int8 quantization.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized values in int8.
    pub data: Vec<i8>,
    /// Scale factor: `float_value ≈ data[i] * scale`.
    pub scale: f32,
}

/// Result of int4 (nibble-packed) quantization.
#[derive(Debug, Clone)]
pub struct QuantizedTensorI4 {
    /// Packed nibbles – two int4 values per byte (low nibble first).
    pub data: Vec<u8>,
    /// Scale factor: `float_value ≈ unpack(data[i]) * scale`.
    pub scale: f32,
    /// Number of logical elements.
    pub len: usize,
}

/// Quantize an f32 slice to symmetric int8.
pub fn quantize_to_i8(input: &[f32]) -> QuantizedTensor {
    if input.is_empty() {
        return QuantizedTensor { data: Vec::new(), scale: 0.0 };
    }
    let abs_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if abs_max == 0.0 {
        return QuantizedTensor { data: vec![0i8; input.len()], scale: 0.0 };
    }
    let scale = abs_max / 127.0;
    let inv_scale = 127.0 / abs_max;
    let data = input.iter().map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8).collect();
    QuantizedTensor { data, scale }
}

/// Quantize an f32 slice to symmetric int4 (packed nibbles).
pub fn quantize_to_i4(input: &[f32]) -> QuantizedTensorI4 {
    if input.is_empty() {
        return QuantizedTensorI4 { data: Vec::new(), scale: 0.0, len: 0 };
    }
    let abs_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if abs_max == 0.0 {
        return QuantizedTensorI4 {
            data: vec![0u8; input.len().div_ceil(2)],
            scale: 0.0,
            len: input.len(),
        };
    }
    let scale = abs_max / 7.0;
    let inv_scale = 7.0 / abs_max;
    let quantized: Vec<i8> =
        input.iter().map(|&v| (v * inv_scale).round().clamp(-7.0, 7.0) as i8).collect();
    let packed_len = quantized.len().div_ceil(2);
    let mut data = vec![0u8; packed_len];
    for (i, chunk) in quantized.chunks(2).enumerate() {
        let lo = (chunk[0] & 0x0F) as u8;
        let hi = if chunk.len() > 1 { (chunk[1] & 0x0F) as u8 } else { 0 };
        data[i] = lo | (hi << 4);
    }
    QuantizedTensorI4 { data, scale, len: input.len() }
}

/// Unpack int4 nibbles to int8.
pub fn unpack_i4_to_i8(packed: &QuantizedTensorI4) -> Vec<i8> {
    let mut out = Vec::with_capacity(packed.len);
    for &byte in &packed.data {
        let lo = (byte & 0x0F) as i8;
        // Sign-extend from 4-bit.
        let lo = if lo > 7 { lo - 16 } else { lo };
        out.push(lo);
        if out.len() < packed.len {
            let hi = ((byte >> 4) & 0x0F) as i8;
            let hi = if hi > 7 { hi - 16 } else { hi };
            out.push(hi);
        }
    }
    out.truncate(packed.len);
    out
}

/// Dequantize int8 back to f32.
pub fn dequantize_i8(q: &QuantizedTensor) -> Vec<f32> {
    q.data.iter().map(|&v| v as f32 * q.scale).collect()
}

// ── Runtime dispatch ───────────────────────────────────────────────────

/// Returns `true` when AVX2 is available at runtime.
#[inline]
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── Validation ─────────────────────────────────────────────────────────

fn validate_config(cfg: &QuantizedAttentionConfig) -> Result<()> {
    if cfg.seq_len == 0 || cfg.head_dim == 0 || cfg.num_heads == 0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "QuantizedAttentionConfig: seq_len, head_dim, and num_heads must be > 0".into(),
        }));
    }
    Ok(())
}

fn validate_projection_inputs(
    input: &[f32],
    weight: &[i8],
    weight_scale: f32,
    in_dim: usize,
    out_dim: usize,
) -> Result<()> {
    let needed_input = in_dim;
    let needed_weight = in_dim * out_dim;
    if input.len() < needed_input {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("input length {} < required {}", input.len(), needed_input),
        }));
    }
    if weight.len() < needed_weight {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("weight length {} < required {}", weight.len(), needed_weight),
        }));
    }
    if weight_scale <= 0.0 {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "weight_scale must be positive".into(),
        }));
    }
    Ok(())
}

// ── 1. Quantized QKV projection ────────────────────────────────────────

/// Project a single-token f32 input through a quantized int8 weight
/// matrix, producing an f32 output.
///
/// `output[j] = sum_i(input[i] * weight[j * in_dim + i]) * weight_scale`
///
/// Weight layout is row-major: `[out_dim, in_dim]`.
pub fn quantized_qkv_projection(
    input: &[f32],
    weight: &[i8],
    weight_scale: f32,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    validate_projection_inputs(input, weight, weight_scale, in_dim, out_dim)?;
    if output.len() < out_dim {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output length {} < required {}", output.len(), out_dim),
        }));
    }
    if has_avx2() {
        // Safety: feature detection guarantees AVX2 availability.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            qkv_projection_avx2(input, weight, weight_scale, in_dim, out_dim, output);
            return Ok(());
        }
    }
    qkv_projection_scalar(input, weight, weight_scale, in_dim, out_dim, output);
    Ok(())
}

fn qkv_projection_scalar(
    input: &[f32],
    weight: &[i8],
    weight_scale: f32,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    for j in 0..out_dim {
        let row = &weight[j * in_dim..(j + 1) * in_dim];
        let acc: f32 =
            input[..in_dim].iter().zip(row.iter()).map(|(&inp, &w)| inp * w as f32).sum();
        output[j] = acc * weight_scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn qkv_projection_avx2(
    input: &[f32],
    weight: &[i8],
    weight_scale: f32,
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    for j in 0..out_dim {
        let row = &weight[j * in_dim..(j + 1) * in_dim];
        let mut acc = _mm256_setzero_ps();
        let chunks = in_dim / 8;
        for c in 0..chunks {
            let base = c * 8;
            // Load 8 int8 weights, convert to f32, multiply-accumulate.
            let w_vals: [f32; 8] = std::array::from_fn(|k| row[base + k] as f32);
            let w_vec = _mm256_loadu_ps(w_vals.as_ptr());
            let i_vec = _mm256_loadu_ps(input.as_ptr().add(base));
            acc = _mm256_fmadd_ps(w_vec, i_vec, acc);
        }
        // Horizontal sum of the 8 lanes.
        let mut sum = hsum_avx2(acc);
        // Scalar tail.
        for (&inp, &w) in input[chunks * 8..in_dim].iter().zip(row[chunks * 8..in_dim].iter()) {
            sum += inp * w as f32;
        }
        output[j] = sum * weight_scale;
    }
}

/// Horizontal sum of an `__m256`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// ── 2. Quantized score computation ─────────────────────────────────────

/// Compute attention scores using integer dot products.
///
/// `scores[i][j] = dot(q_row_i, k_row_j) * q_scale * k_scale * attn_scale`
///
/// Both `q` and `k` are `[seq_len, head_dim]` in row-major order.
pub fn quantized_score_computation(
    q: &QuantizedTensor,
    k: &QuantizedTensor,
    seq_len: usize,
    head_dim: usize,
    attn_scale: f32,
    scores: &mut [f32],
) -> Result<()> {
    let total = seq_len * head_dim;
    if q.data.len() < total || k.data.len() < total {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "q/k tensor too small for seq_len * head_dim".into(),
        }));
    }
    if scores.len() < seq_len * seq_len {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "scores buffer too small for seq_len * seq_len".into(),
        }));
    }
    let combined_scale = q.scale * k.scale * attn_scale;

    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            score_computation_avx2(&q.data, &k.data, seq_len, head_dim, combined_scale, scores);
            return Ok(());
        }
    }
    score_computation_scalar(&q.data, &k.data, seq_len, head_dim, combined_scale, scores);
    Ok(())
}

fn score_computation_scalar(
    q: &[i8],
    k: &[i8],
    seq_len: usize,
    head_dim: usize,
    combined_scale: f32,
    scores: &mut [f32],
) {
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot: i32 = q_row
                .iter()
                .zip(k_row.iter())
                .map(|(&q_val, &k_val)| q_val as i32 * k_val as i32)
                .sum();
            scores[i * seq_len + j] = dot as f32 * combined_scale;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn score_computation_avx2(
    q: &[i8],
    k: &[i8],
    seq_len: usize,
    head_dim: usize,
    combined_scale: f32,
    scores: &mut [f32],
) {
    for i in 0..seq_len {
        let q_base = i * head_dim;
        for j in 0..seq_len {
            let k_base = j * head_dim;
            let mut acc = _mm256_setzero_si256();
            let chunks = head_dim / 16;
            for c in 0..chunks {
                let offset = c * 16;
                let q_vec = _mm_loadu_si128(q.as_ptr().add(q_base + offset) as *const __m128i);
                let k_vec = _mm_loadu_si128(k.as_ptr().add(k_base + offset) as *const __m128i);
                // _mm256_cvtepi8_epi16 sign-extends 16 × i8 → 16 × i16
                let q16 = _mm256_cvtepi8_epi16(q_vec);
                let k16 = _mm256_cvtepi8_epi16(k_vec);
                // _mm256_madd_epi16: pairs of i16 multiplied and horizontally added → i32
                let prod = _mm256_madd_epi16(q16, k16);
                acc = _mm256_add_epi32(acc, prod);
            }
            // Horizontal i32 sum.
            let dot = hsum_epi32_avx2(acc);
            // Scalar tail.
            let mut tail_dot = 0i32;
            for d in (chunks * 16)..head_dim {
                tail_dot += q[q_base + d] as i32 * k[k_base + d] as i32;
            }
            scores[i * seq_len + j] = (dot + tail_dot) as f32 * combined_scale;
        }
    }
}

/// Horizontal sum of i32 lanes in a `__m256i`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0b_01_00_11_10);
    let sums = _mm_add_epi32(sum128, shuf);
    let shuf2 = _mm_shuffle_epi32(sums, 0b_00_00_00_01);
    let sums2 = _mm_add_epi32(sums, shuf2);
    _mm_cvtsi128_si32(sums2)
}

// ── 3. Quantized softmax ───────────────────────────────────────────────

/// Fixed-point softmax approximation operating on f32 scores in-place.
///
/// Applies the standard numerically-stable softmax:
///   `softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))`
///
/// with optional causal masking (set future positions to `-inf`).
pub fn quantized_softmax(scores: &mut [f32], seq_len: usize, causal: bool) -> Result<()> {
    if scores.len() < seq_len * seq_len {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "scores buffer too small for seq_len * seq_len".into(),
        }));
    }
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        // Apply causal mask.
        if causal {
            for elem in row.iter_mut().skip(i + 1) {
                *elem = f32::NEG_INFINITY;
            }
        }
        if has_avx2() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                softmax_row_avx2(row);
                continue;
            }
        }
        softmax_row_scalar(row);
    }
    Ok(())
}

fn softmax_row_scalar(row: &mut [f32]) {
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        // Entire row masked – set uniform zero.
        row.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        row.iter_mut().for_each(|v| *v *= inv);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_row_avx2(row: &mut [f32]) {
    let n = row.len();
    // Find max.
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let chunks = n / 8;
    for c in 0..chunks {
        let v = _mm256_loadu_ps(row.as_ptr().add(c * 8));
        max_vec = _mm256_max_ps(max_vec, v);
    }
    let mut max_val = hsum_max_avx2(max_vec);
    for val in &row[chunks * 8..n] {
        max_val = max_val.max(*val);
    }
    if max_val == f32::NEG_INFINITY {
        row.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    // exp(x - max) and sum.
    let max_broadcast = _mm256_set1_ps(max_val);
    let mut sum_vec = _mm256_setzero_ps();
    for c in 0..chunks {
        let ptr = row.as_mut_ptr().add(c * 8);
        let v = _mm256_loadu_ps(ptr);
        let shifted = _mm256_sub_ps(v, max_broadcast);
        // Use scalar exp per-element (fast_exp could be added later).
        let mut arr = [0.0f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), shifted);
        for x in &mut arr {
            *x = x.exp();
        }
        let exp_v = _mm256_loadu_ps(arr.as_ptr());
        _mm256_storeu_ps(ptr, exp_v);
        sum_vec = _mm256_add_ps(sum_vec, exp_v);
    }
    let mut sum = hsum_avx2(sum_vec);
    for val in &mut row[chunks * 8..n] {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    // Normalize.
    if sum > 0.0 {
        let inv = 1.0 / sum;
        let inv_vec = _mm256_set1_ps(inv);
        for c in 0..chunks {
            let ptr = row.as_mut_ptr().add(c * 8);
            let v = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, _mm256_mul_ps(v, inv_vec));
        }
        for val in &mut row[chunks * 8..n] {
            *val *= inv;
        }
    }
}

/// Horizontal max of an `__m256`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_max_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(m64, m64);
    let m32 = _mm_max_ss(m64, shuf2);
    _mm_cvtss_f32(m32)
}

// ── 4. Quantized value aggregation ─────────────────────────────────────

/// Weighted sum in the quantized domain.
///
/// `output[i, d] = sum_j(weights[i, j] * v[j, d]) * v_scale`
///
/// `weights` are f32 attention probabilities `[seq_len, seq_len]`,
/// `v` is a `QuantizedTensor` of shape `[seq_len, head_dim]`.
pub fn quantized_value_aggregation(
    weights: &[f32],
    v: &QuantizedTensor,
    seq_len: usize,
    head_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    let total = seq_len * head_dim;
    if v.data.len() < total {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "v tensor too small for seq_len * head_dim".into(),
        }));
    }
    if weights.len() < seq_len * seq_len {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "weights buffer too small".into(),
        }));
    }
    if output.len() < total {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output length {} < required {}", output.len(), total),
        }));
    }
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            value_aggregation_avx2(weights, &v.data, v.scale, seq_len, head_dim, output);
            return Ok(());
        }
    }
    value_aggregation_scalar(weights, &v.data, v.scale, seq_len, head_dim, output);
    Ok(())
}

fn value_aggregation_scalar(
    weights: &[f32],
    v: &[i8],
    v_scale: f32,
    seq_len: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    for i in 0..seq_len {
        let w_row = &weights[i * seq_len..(i + 1) * seq_len];
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for (j, &w) in w_row[..seq_len].iter().enumerate() {
                acc += w * v[j * head_dim + d] as f32;
            }
            output[i * head_dim + d] = acc * v_scale;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn value_aggregation_avx2(
    weights: &[f32],
    v: &[i8],
    v_scale: f32,
    seq_len: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    for i in 0..seq_len {
        let w_row = &weights[i * seq_len..(i + 1) * seq_len];
        for d in 0..head_dim {
            let mut acc = _mm256_setzero_ps();
            let chunks = seq_len / 8;
            for c in 0..chunks {
                let base = c * 8;
                let w_vec = _mm256_loadu_ps(w_row.as_ptr().add(base));
                let v_vals: [f32; 8] = std::array::from_fn(|k| v[(base + k) * head_dim + d] as f32);
                let v_vec = _mm256_loadu_ps(v_vals.as_ptr());
                acc = _mm256_fmadd_ps(w_vec, v_vec, acc);
            }
            let mut sum = hsum_avx2(acc);
            for j in (chunks * 8)..seq_len {
                sum += w_row[j] * v[j * head_dim + d] as f32;
            }
            output[i * head_dim + d] = sum * v_scale;
        }
    }
}

// ── 5. Dequantized output ──────────────────────────────────────────────

/// Convert a quantized int8 attention result back to f32.
///
/// `output[i] = quantized[i] * scale + bias[i]`  (bias is optional).
pub fn dequantized_output(
    quantized: &[i8],
    scale: f32,
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    let n = quantized.len();
    if output.len() < n {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output length {} < required {}", output.len(), n),
        }));
    }
    if let Some(b) = bias
        && b.len() < n
    {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: "bias length too small".into(),
        }));
    }
    if has_avx2() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            dequantized_output_avx2(quantized, scale, bias, output);
            return Ok(());
        }
    }
    dequantized_output_scalar(quantized, scale, bias, output);
    Ok(())
}

fn dequantized_output_scalar(
    quantized: &[i8],
    scale: f32,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    for (i, &q) in quantized.iter().enumerate() {
        let val = q as f32 * scale;
        output[i] = if let Some(b) = bias { val + b[i] } else { val };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantized_output_avx2(
    quantized: &[i8],
    scale: f32,
    bias: Option<&[f32]>,
    output: &mut [f32],
) {
    let n = quantized.len();
    let scale_vec = _mm256_set1_ps(scale);
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let q_vals: [f32; 8] = std::array::from_fn(|k| quantized[base + k] as f32);
        let q_vec = _mm256_loadu_ps(q_vals.as_ptr());
        let mut result = _mm256_mul_ps(q_vec, scale_vec);
        if let Some(b) = bias {
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(base));
            result = _mm256_add_ps(result, b_vec);
        }
        _mm256_storeu_ps(output.as_mut_ptr().add(base), result);
    }
    for (idx, (&q, out)) in
        quantized[chunks * 8..n].iter().zip(output[chunks * 8..n].iter_mut()).enumerate()
    {
        let val = q as f32 * scale;
        *out = if let Some(b) = bias { val + b[chunks * 8 + idx] } else { val };
    }
}

// ── 6. Full quantized self-attention pipeline ──────────────────────────

/// End-to-end quantized self-attention.
///
/// Takes f32 input `[seq_len, head_dim]` per head, quantizes Q/K/V
/// through provided int8 weight matrices, computes scaled dot-product
/// attention in the quantized domain, and returns f32 output.
///
/// Weight matrices are `[head_dim, head_dim]` row-major int8 with a
/// per-matrix scale factor.
#[allow(clippy::too_many_arguments)]
pub fn quantized_self_attention(
    input: &[f32],
    wq: &[i8],
    wq_scale: f32,
    wk: &[i8],
    wk_scale: f32,
    wv: &[i8],
    wv_scale: f32,
    cfg: &QuantizedAttentionConfig,
    output: &mut [f32],
) -> Result<()> {
    validate_config(cfg)?;
    let seq_len = cfg.seq_len;
    let head_dim = cfg.head_dim;
    let total = seq_len * head_dim;

    if input.len() < total {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("input length {} < required {}", input.len(), total),
        }));
    }
    if output.len() < total {
        return Err(BitNetError::Kernel(KernelError::ExecutionFailed {
            reason: format!("output length {} < required {}", output.len(), total),
        }));
    }

    // 1. QKV projections.
    let mut q_proj = vec![0.0f32; total];
    let mut k_proj = vec![0.0f32; total];
    let mut v_proj = vec![0.0f32; total];
    for t in 0..seq_len {
        let in_row = &input[t * head_dim..(t + 1) * head_dim];
        quantized_qkv_projection(
            in_row,
            wq,
            wq_scale,
            head_dim,
            head_dim,
            &mut q_proj[t * head_dim..(t + 1) * head_dim],
        )?;
        quantized_qkv_projection(
            in_row,
            wk,
            wk_scale,
            head_dim,
            head_dim,
            &mut k_proj[t * head_dim..(t + 1) * head_dim],
        )?;
        quantized_qkv_projection(
            in_row,
            wv,
            wv_scale,
            head_dim,
            head_dim,
            &mut v_proj[t * head_dim..(t + 1) * head_dim],
        )?;
    }

    // 2. Quantize Q and K for integer score computation.
    let q_quant = quantize_to_i8(&q_proj);
    let k_quant = quantize_to_i8(&k_proj);

    // 3. Score computation.
    let attn_scale = cfg.effective_scale();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    quantized_score_computation(&q_quant, &k_quant, seq_len, head_dim, attn_scale, &mut scores)?;

    // 4. Softmax.
    quantized_softmax(&mut scores, seq_len, cfg.causal)?;

    // 5. Value aggregation.
    let v_quant = quantize_to_i8(&v_proj);
    quantized_value_aggregation(&scores, &v_quant, seq_len, head_dim, output)?;

    Ok(())
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn vec_approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| approx_eq(x, y, tol))
    }

    fn identity_i8(dim: usize) -> (Vec<i8>, f32) {
        let mut w = vec![0i8; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 127;
        }
        (w, 1.0 / 127.0)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    // ── QuantizedAttentionConfig tests ─────────────────────────────────

    #[test]
    fn test_config_default_scale() {
        let cfg = QuantizedAttentionConfig::new(4, 64, 8);
        let expected = 1.0 / 64.0f32.sqrt();
        assert!(approx_eq(cfg.effective_scale(), expected, 1e-6));
    }

    #[test]
    fn test_config_custom_scale() {
        let mut cfg = QuantizedAttentionConfig::new(4, 64, 8);
        cfg.scale = Some(0.5);
        assert_eq!(cfg.effective_scale(), 0.5);
    }

    #[test]
    fn test_config_causal_default_false() {
        let cfg = QuantizedAttentionConfig::new(4, 64, 8);
        assert!(!cfg.causal);
    }

    #[test]
    fn test_validate_config_zero_seq_len() {
        let cfg = QuantizedAttentionConfig::new(0, 64, 8);
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_config_zero_head_dim() {
        let cfg = QuantizedAttentionConfig::new(4, 0, 8);
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_config_zero_num_heads() {
        let cfg = QuantizedAttentionConfig::new(4, 64, 0);
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn test_validate_config_valid() {
        let cfg = QuantizedAttentionConfig::new(4, 64, 8);
        assert!(validate_config(&cfg).is_ok());
    }

    // ── Quantize / dequantize round-trip tests ─────────────────────────

    #[test]
    fn test_quantize_i8_round_trip() {
        let input = vec![1.0, -0.5, 0.25, 0.0, -1.0];
        let qt = quantize_to_i8(&input);
        let deq = dequantize_i8(&qt);
        assert!(vec_approx_eq(&input, &deq, 0.02));
    }

    #[test]
    fn test_quantize_i8_empty() {
        let qt = quantize_to_i8(&[]);
        assert!(qt.data.is_empty());
        assert_eq!(qt.scale, 0.0);
    }

    #[test]
    fn test_quantize_i8_all_zeros() {
        let qt = quantize_to_i8(&[0.0, 0.0, 0.0]);
        assert_eq!(qt.scale, 0.0);
        assert!(qt.data.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_i8_single_value() {
        let qt = quantize_to_i8(&[3.14]);
        assert_eq!(qt.data.len(), 1);
        assert_eq!(qt.data[0], 127);
    }

    #[test]
    fn test_quantize_i8_symmetric_range() {
        let input = vec![-1.0, 1.0];
        let qt = quantize_to_i8(&input);
        assert_eq!(qt.data[0], -127);
        assert_eq!(qt.data[1], 127);
    }

    #[test]
    fn test_quantize_i8_preserves_sign() {
        let input = vec![-0.5, 0.5, -0.1, 0.1];
        let qt = quantize_to_i8(&input);
        assert!(qt.data[0] < 0);
        assert!(qt.data[1] > 0);
        assert!(qt.data[2] < 0);
        assert!(qt.data[3] > 0);
    }

    #[test]
    fn test_quantize_i4_round_trip() {
        let input = vec![1.0, -0.5, 0.25, 0.0];
        let qt = quantize_to_i4(&input);
        let unpacked = unpack_i4_to_i8(&qt);
        let deq: Vec<f32> = unpacked.iter().map(|&v| v as f32 * qt.scale).collect();
        assert!(vec_approx_eq(&input, &deq, 0.2));
    }

    #[test]
    fn test_quantize_i4_empty() {
        let qt = quantize_to_i4(&[]);
        assert!(qt.data.is_empty());
        assert_eq!(qt.len, 0);
    }

    #[test]
    fn test_quantize_i4_all_zeros() {
        let qt = quantize_to_i4(&[0.0, 0.0]);
        assert_eq!(qt.scale, 0.0);
        assert_eq!(qt.len, 2);
    }

    #[test]
    fn test_quantize_i4_odd_length() {
        let input = vec![1.0, -1.0, 0.5];
        let qt = quantize_to_i4(&input);
        assert_eq!(qt.len, 3);
        let unpacked = unpack_i4_to_i8(&qt);
        assert_eq!(unpacked.len(), 3);
    }

    #[test]
    fn test_unpack_i4_sign_extension() {
        let packed = QuantizedTensorI4 { data: vec![0xF0], scale: 1.0, len: 2 };
        let unpacked = unpack_i4_to_i8(&packed);
        assert_eq!(unpacked[0], 0); // low nibble = 0
        assert_eq!(unpacked[1], -1); // high nibble = 0xF → -1
    }

    #[test]
    fn test_dequantize_i8_simple() {
        let qt = QuantizedTensor { data: vec![127, -127, 0], scale: 0.01 };
        let deq = dequantize_i8(&qt);
        assert!(approx_eq(deq[0], 1.27, 1e-6));
        assert!(approx_eq(deq[1], -1.27, 1e-6));
        assert!(approx_eq(deq[2], 0.0, 1e-6));
    }

    // ── QKV projection tests ──────────────────────────────────────────

    #[test]
    fn test_qkv_projection_identity() {
        let dim = 16;
        let (w, ws) = identity_i8(dim);
        let input: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; dim];
        quantized_qkv_projection(&input, &w, ws, dim, dim, &mut output).unwrap();
        assert!(vec_approx_eq(&input, &output, 0.02));
    }

    #[test]
    fn test_qkv_projection_zero_input() {
        let dim = 8;
        let (w, ws) = identity_i8(dim);
        let input = vec![0.0f32; dim];
        let mut output = vec![1.0f32; dim];
        quantized_qkv_projection(&input, &w, ws, dim, dim, &mut output).unwrap();
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_qkv_projection_non_square() {
        let in_dim = 4;
        let out_dim = 8;
        let weight = vec![1i8; in_dim * out_dim];
        let input = vec![1.0f32; in_dim];
        let mut output = vec![0.0f32; out_dim];
        quantized_qkv_projection(&input, &weight, 1.0, in_dim, out_dim, &mut output).unwrap();
        // Each output = sum of 4 × (1.0 * 1) * 1.0 = 4.0
        assert!(output.iter().all(|&v| approx_eq(v, 4.0, 1e-4)));
    }

    #[test]
    fn test_qkv_projection_negative_weights() {
        let in_dim = 4;
        let out_dim = 2;
        let weight = vec![-1i8; in_dim * out_dim];
        let input = vec![1.0f32; in_dim];
        let mut output = vec![0.0f32; out_dim];
        quantized_qkv_projection(&input, &weight, 1.0, in_dim, out_dim, &mut output).unwrap();
        assert!(output.iter().all(|&v| approx_eq(v, -4.0, 1e-4)));
    }

    #[test]
    fn test_qkv_projection_input_too_small() {
        let w = vec![1i8; 16];
        let mut out = vec![0.0f32; 4];
        let res = quantized_qkv_projection(&[1.0, 2.0], &w, 1.0, 4, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_qkv_projection_weight_too_small() {
        let mut out = vec![0.0f32; 4];
        let res = quantized_qkv_projection(&[1.0; 4], &[1i8; 8], 1.0, 4, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_qkv_projection_output_too_small() {
        let mut out = vec![0.0f32; 2];
        let res = quantized_qkv_projection(&[1.0; 4], &[1i8; 16], 1.0, 4, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_qkv_projection_negative_scale() {
        let mut out = vec![0.0f32; 4];
        let res = quantized_qkv_projection(&[1.0; 4], &[1i8; 16], -1.0, 4, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_qkv_projection_large_dim() {
        let dim = 128;
        let (w, ws) = identity_i8(dim);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let mut output = vec![0.0f32; dim];
        quantized_qkv_projection(&input, &w, ws, dim, dim, &mut output).unwrap();
        assert!(vec_approx_eq(&input, &output, 0.02));
    }

    // ── Score computation tests ────────────────────────────────────────

    #[test]
    fn test_score_computation_identity() {
        let seq_len = 2;
        let head_dim = 4;
        let q = QuantizedTensor { data: vec![127, 0, 0, 0, 0, 127, 0, 0], scale: 1.0 / 127.0 };
        let k = QuantizedTensor { data: vec![127, 0, 0, 0, 0, 127, 0, 0], scale: 1.0 / 127.0 };
        let mut scores = vec![0.0f32; 4];
        quantized_score_computation(&q, &k, seq_len, head_dim, 1.0, &mut scores).unwrap();
        // Diagonal should be ~1.0, off-diagonal ~0.0
        assert!(approx_eq(scores[0], 1.0, 0.02));
        assert!(approx_eq(scores[3], 1.0, 0.02));
        assert!(approx_eq(scores[1], 0.0, 0.02));
        assert!(approx_eq(scores[2], 0.0, 0.02));
    }

    #[test]
    fn test_score_computation_all_same() {
        let seq_len = 3;
        let head_dim = 4;
        let q = QuantizedTensor { data: vec![10i8; 12], scale: 0.1 };
        let k = QuantizedTensor { data: vec![10i8; 12], scale: 0.1 };
        let mut scores = vec![0.0f32; 9];
        quantized_score_computation(&q, &k, seq_len, head_dim, 1.0, &mut scores).unwrap();
        // All scores should be identical.
        let first = scores[0];
        assert!(scores.iter().all(|&s| approx_eq(s, first, 1e-6)));
    }

    #[test]
    fn test_score_computation_scale_factor() {
        let seq_len = 1;
        let head_dim = 4;
        let q = QuantizedTensor { data: vec![10, 10, 10, 10], scale: 1.0 };
        let k = QuantizedTensor { data: vec![10, 10, 10, 10], scale: 1.0 };
        let mut scores = vec![0.0f32; 1];
        quantized_score_computation(&q, &k, seq_len, head_dim, 0.5, &mut scores).unwrap();
        // dot = 4 × 100 = 400, combined = 400 * 1.0 * 1.0 * 0.5 = 200.0
        assert!(approx_eq(scores[0], 200.0, 1e-3));
    }

    #[test]
    fn test_score_computation_q_too_small() {
        let q = QuantizedTensor { data: vec![1; 3], scale: 1.0 };
        let k = QuantizedTensor { data: vec![1; 8], scale: 1.0 };
        let mut scores = vec![0.0f32; 4];
        let res = quantized_score_computation(&q, &k, 2, 4, 1.0, &mut scores);
        assert!(res.is_err());
    }

    #[test]
    fn test_score_computation_scores_too_small() {
        let q = QuantizedTensor { data: vec![1; 8], scale: 1.0 };
        let k = QuantizedTensor { data: vec![1; 8], scale: 1.0 };
        let mut scores = vec![0.0f32; 2];
        let res = quantized_score_computation(&q, &k, 2, 4, 1.0, &mut scores);
        assert!(res.is_err());
    }

    #[test]
    fn test_score_computation_orthogonal() {
        let q = QuantizedTensor { data: vec![127, 0, 0, 127], scale: 1.0 / 127.0 };
        let k = QuantizedTensor { data: vec![0, 127, 127, 0], scale: 1.0 / 127.0 };
        let mut scores = vec![0.0f32; 4];
        quantized_score_computation(&q, &k, 2, 2, 1.0, &mut scores).unwrap();
        assert!(approx_eq(scores[0 * 2 + 0], 0.0, 0.02)); // q0 · k0
        assert!(approx_eq(scores[1 * 2 + 1], 0.0, 0.02)); // q1 · k1
    }

    #[test]
    fn test_score_computation_large_head_dim() {
        let seq_len = 2;
        let head_dim = 64;
        let q = QuantizedTensor { data: vec![1i8; seq_len * head_dim], scale: 0.01 };
        let k = QuantizedTensor { data: vec![1i8; seq_len * head_dim], scale: 0.01 };
        let mut scores = vec![0.0f32; seq_len * seq_len];
        quantized_score_computation(&q, &k, seq_len, head_dim, 1.0, &mut scores).unwrap();
        // dot = 64, combined = 64 * 0.01 * 0.01 * 1.0 = 0.0064
        let expected = 64.0 * 0.0001;
        assert!(scores.iter().all(|&s| approx_eq(s, expected, 1e-5)));
    }

    // ── Softmax tests ──────────────────────────────────────────────────

    #[test]
    fn test_softmax_uniform() {
        let mut scores = vec![1.0f32; 4];
        quantized_softmax(&mut scores, 2, false).unwrap();
        // Each row should be [0.5, 0.5].
        assert!(approx_eq(scores[0], 0.5, 1e-6));
        assert!(approx_eq(scores[1], 0.5, 1e-6));
    }

    #[test]
    fn test_softmax_single_element() {
        let mut scores = vec![42.0f32];
        quantized_softmax(&mut scores, 1, false).unwrap();
        assert!(approx_eq(scores[0], 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_row_sums_to_one() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        quantized_softmax(&mut scores, 3, false).unwrap();
        for row in 0..3 {
            let sum: f32 = scores[row * 3..(row + 1) * 3].iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-5));
        }
    }

    #[test]
    fn test_softmax_large_values() {
        let mut scores = vec![1000.0, 1001.0, 1000.0, 999.0];
        quantized_softmax(&mut scores, 2, false).unwrap();
        let sum0: f32 = scores[0..2].iter().sum();
        let sum1: f32 = scores[2..4].iter().sum();
        assert!(approx_eq(sum0, 1.0, 1e-5));
        assert!(approx_eq(sum1, 1.0, 1e-5));
    }

    #[test]
    fn test_softmax_negative_values() {
        let mut scores = vec![-1.0, -2.0, -3.0, -4.0];
        quantized_softmax(&mut scores, 2, false).unwrap();
        let sum: f32 = scores[0..2].iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));
    }

    #[test]
    fn test_softmax_causal_mask() {
        let mut scores = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        quantized_softmax(&mut scores, 3, true).unwrap();
        // Row 0: only [0] visible → 1.0.
        assert!(approx_eq(scores[0], 1.0, 1e-5));
        assert!(approx_eq(scores[1], 0.0, 1e-5));
        assert!(approx_eq(scores[2], 0.0, 1e-5));
        // Row 1: [0..=1] visible → 0.5 each.
        assert!(approx_eq(scores[3], 0.5, 1e-5));
        assert!(approx_eq(scores[4], 0.5, 1e-5));
        assert!(approx_eq(scores[5], 0.0, 1e-5));
        // Row 2: all visible → 1/3 each.
        assert!(approx_eq(scores[6], 1.0 / 3.0, 1e-5));
    }

    #[test]
    fn test_softmax_causal_single_row() {
        let mut scores = vec![5.0];
        quantized_softmax(&mut scores, 1, true).unwrap();
        assert!(approx_eq(scores[0], 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_monotonicity() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        quantized_softmax(&mut scores, 2, false).unwrap();
        // Within each row, larger input → larger prob.
        assert!(scores[1] > scores[0]);
        assert!(scores[3] > scores[2]);
    }

    #[test]
    fn test_softmax_buffer_too_small() {
        let mut scores = vec![1.0; 3];
        let res = quantized_softmax(&mut scores, 2, false);
        assert!(res.is_err());
    }

    #[test]
    fn test_softmax_all_neg_inf() {
        let mut scores = vec![f32::NEG_INFINITY; 4];
        quantized_softmax(&mut scores, 2, false).unwrap();
        // Should produce zeros (degenerate case).
        assert!(scores.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_softmax_non_power_of_two() {
        let seq = 5;
        let mut scores = vec![1.0f32; seq * seq];
        quantized_softmax(&mut scores, seq, false).unwrap();
        for row in 0..seq {
            let sum: f32 = scores[row * seq..(row + 1) * seq].iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-5));
        }
    }

    // ── Value aggregation tests ────────────────────────────────────────

    #[test]
    fn test_value_aggregation_identity_weights() {
        // Identity attention (weight matrix = I).
        let seq = 2;
        let hd = 4;
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity
        let v = QuantizedTensor { data: vec![10, 20, 30, 40, 50, 60, 70, 80], scale: 0.1 };
        let mut output = vec![0.0f32; seq * hd];
        quantized_value_aggregation(&weights, &v, seq, hd, &mut output).unwrap();
        // Row 0 = v[0] * 0.1, row 1 = v[1] * 0.1.
        assert!(approx_eq(output[0], 1.0, 1e-4));
        assert!(approx_eq(output[4], 5.0, 1e-4));
    }

    #[test]
    fn test_value_aggregation_uniform_weights() {
        let seq = 2;
        let hd = 2;
        let weights = vec![0.5, 0.5, 0.5, 0.5];
        let v = QuantizedTensor { data: vec![10, 20, 30, 40], scale: 1.0 };
        let mut output = vec![0.0f32; seq * hd];
        quantized_value_aggregation(&weights, &v, seq, hd, &mut output).unwrap();
        // Each output = avg of rows.
        assert!(approx_eq(output[0], 20.0, 1e-3)); // 0.5*10 + 0.5*30
        assert!(approx_eq(output[1], 30.0, 1e-3)); // 0.5*20 + 0.5*40
    }

    #[test]
    fn test_value_aggregation_v_too_small() {
        let v = QuantizedTensor { data: vec![1; 3], scale: 1.0 };
        let mut out = vec![0.0; 8];
        let res = quantized_value_aggregation(&[0.5; 4], &v, 2, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_value_aggregation_output_too_small() {
        let v = QuantizedTensor { data: vec![1; 8], scale: 1.0 };
        let mut out = vec![0.0; 2];
        let res = quantized_value_aggregation(&[0.5; 4], &v, 2, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_value_aggregation_weights_too_small() {
        let v = QuantizedTensor { data: vec![1; 8], scale: 1.0 };
        let mut out = vec![0.0; 8];
        let res = quantized_value_aggregation(&[0.5; 2], &v, 2, 4, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_value_aggregation_zero_weights() {
        let seq = 2;
        let hd = 4;
        let weights = vec![0.0f32; seq * seq];
        let v = QuantizedTensor { data: vec![100i8; seq * hd], scale: 1.0 };
        let mut output = vec![99.0f32; seq * hd];
        quantized_value_aggregation(&weights, &v, seq, hd, &mut output).unwrap();
        assert!(output.iter().all(|&v| approx_eq(v, 0.0, 1e-6)));
    }

    // ── Dequantized output tests ───────────────────────────────────────

    #[test]
    fn test_dequantized_output_no_bias() {
        let q = vec![127i8, -127, 0, 64];
        let scale = 0.01;
        let mut out = vec![0.0f32; 4];
        dequantized_output(&q, scale, None, &mut out).unwrap();
        assert!(approx_eq(out[0], 1.27, 1e-5));
        assert!(approx_eq(out[1], -1.27, 1e-5));
        assert!(approx_eq(out[2], 0.0, 1e-5));
        assert!(approx_eq(out[3], 0.64, 1e-5));
    }

    #[test]
    fn test_dequantized_output_with_bias() {
        let q = vec![100i8, -50];
        let scale = 0.1;
        let bias = vec![1.0f32, 2.0];
        let mut out = vec![0.0f32; 2];
        dequantized_output(&q, scale, Some(&bias), &mut out).unwrap();
        assert!(approx_eq(out[0], 11.0, 1e-4)); // 100*0.1 + 1.0
        assert!(approx_eq(out[1], -3.0, 1e-4)); // -50*0.1 + 2.0
    }

    #[test]
    fn test_dequantized_output_empty() {
        let mut out = vec![0.0f32; 0];
        dequantized_output(&[], 1.0, None, &mut out).unwrap();
    }

    #[test]
    fn test_dequantized_output_output_too_small() {
        let mut out = vec![0.0f32; 1];
        let res = dequantized_output(&[1i8, 2], 1.0, None, &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_dequantized_output_bias_too_small() {
        let mut out = vec![0.0f32; 2];
        let res = dequantized_output(&[1i8, 2], 1.0, Some(&[1.0]), &mut out);
        assert!(res.is_err());
    }

    #[test]
    fn test_dequantized_output_zero_scale() {
        let q = vec![127i8, -127];
        let mut out = vec![99.0f32; 2];
        dequantized_output(&q, 0.0, None, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantized_output_large_batch() {
        let n = 256;
        let q: Vec<i8> = (0..n).map(|i| ((i % 255) as i32 - 127) as i8).collect();
        let mut out = vec![0.0f32; n];
        dequantized_output(&q, 0.01, None, &mut out).unwrap();
        for (i, &v) in out.iter().enumerate() {
            let expected = ((i % 255) as i32 - 127) as f32 * 0.01;
            assert!(approx_eq(v, expected, 1e-5));
        }
    }

    // ── Full self-attention pipeline tests ─────────────────────────────

    #[test]
    fn test_self_attention_identity_weights() {
        let seq_len = 2;
        let head_dim = 4;
        let (w, ws) = identity_i8(head_dim);
        let input: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.05).collect();
        let cfg = QuantizedAttentionConfig::new(seq_len, head_dim, 1);
        let mut output = vec![0.0f32; seq_len * head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        // With identity projections, attention should approximate input.
        let cos = cosine_similarity(&input, &output);
        assert!(cos > 0.85, "cosine similarity too low: {cos}");
    }

    #[test]
    fn test_self_attention_causal() {
        let seq_len = 4;
        let head_dim = 8;
        let (w, ws) = identity_i8(head_dim);
        let input: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.02).collect();
        let mut cfg = QuantizedAttentionConfig::new(seq_len, head_dim, 1);
        cfg.causal = true;
        let mut output = vec![0.0f32; seq_len * head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        // Output should have non-trivial values.
        let norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "output should be non-zero");
    }

    #[test]
    fn test_self_attention_single_token() {
        let head_dim = 8;
        let (w, ws) = identity_i8(head_dim);
        let input: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let cfg = QuantizedAttentionConfig::new(1, head_dim, 1);
        let mut output = vec![0.0f32; head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        // Single token: softmax is [1.0], output ≈ input.
        let cos = cosine_similarity(&input, &output);
        assert!(cos > 0.90, "cosine similarity too low: {cos}");
    }

    #[test]
    fn test_self_attention_output_size() {
        let seq_len = 3;
        let head_dim = 16;
        let (w, ws) = identity_i8(head_dim);
        let input = vec![0.1f32; seq_len * head_dim];
        let cfg = QuantizedAttentionConfig::new(seq_len, head_dim, 1);
        let mut output = vec![0.0f32; seq_len * head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        assert_eq!(output.len(), seq_len * head_dim);
    }

    #[test]
    fn test_self_attention_input_too_small() {
        let head_dim = 8;
        let (w, ws) = identity_i8(head_dim);
        let input = vec![0.1f32; 4]; // too small for seq_len=2
        let cfg = QuantizedAttentionConfig::new(2, head_dim, 1);
        let mut output = vec![0.0f32; 16];
        let res = quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output);
        assert!(res.is_err());
    }

    #[test]
    fn test_self_attention_output_too_small() {
        let head_dim = 8;
        let (w, ws) = identity_i8(head_dim);
        let input = vec![0.1f32; 16];
        let cfg = QuantizedAttentionConfig::new(2, head_dim, 1);
        let mut output = vec![0.0f32; 4]; // too small
        let res = quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output);
        assert!(res.is_err());
    }

    #[test]
    fn test_self_attention_zero_config() {
        let (w, ws) = identity_i8(4);
        let input = vec![0.1f32; 4];
        let cfg = QuantizedAttentionConfig::new(0, 4, 1);
        let mut output = vec![0.0f32; 4];
        let res = quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output);
        assert!(res.is_err());
    }

    #[test]
    fn test_self_attention_custom_scale() {
        let head_dim = 8;
        let (w, ws) = identity_i8(head_dim);
        let input: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
        let mut cfg = QuantizedAttentionConfig::new(1, head_dim, 1);
        cfg.scale = Some(1.0);
        let mut output = vec![0.0f32; head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        // Should still produce valid output.
        assert!(output.iter().all(|v| v.is_finite()));
    }

    // ── Numerical accuracy / edge case tests ───────────────────────────

    #[test]
    fn test_quantize_i8_snr() {
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let qt = quantize_to_i8(&input);
        let deq = dequantize_i8(&qt);
        let mse: f32 = input.iter().zip(&deq).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
            / input.len() as f32;
        // int8 quantization SNR should be decent.
        assert!(mse < 0.001, "MSE too high: {mse}");
    }

    #[test]
    fn test_quantize_i8_max_abs_error() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.05) - 2.5).collect();
        let qt = quantize_to_i8(&input);
        let deq = dequantize_i8(&qt);
        let max_err: f32 =
            input.iter().zip(&deq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let abs_max = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // Max error should be bounded by half a quantization step.
        assert!(max_err < abs_max / 127.0 + 1e-5, "max abs error too high: {max_err}");
    }

    #[test]
    fn test_softmax_numerical_stability_extreme() {
        // Very large values that would overflow naive exp().
        let mut scores = vec![1e10, 1e10, 1e10, 1e10];
        quantized_softmax(&mut scores, 2, false).unwrap();
        assert!(scores.iter().all(|v| v.is_finite()));
        assert!(approx_eq(scores[0] + scores[1], 1.0, 1e-5));
    }

    #[test]
    fn test_softmax_numerical_stability_mixed() {
        let mut scores = vec![-1e6, 0.0, 1e6, -1e6];
        quantized_softmax(&mut scores, 2, false).unwrap();
        // Row 0: exp(-1e6 - 0) ≈ 0, exp(0 - 0) = 1 → [0, 1].
        assert!(approx_eq(scores[1], 1.0, 1e-5));
        // Row 1: exp(1e6 - 1e6) = 1, exp(-1e6 - 1e6) ≈ 0 → [1, 0].
        assert!(approx_eq(scores[2], 1.0, 1e-5));
    }

    #[test]
    fn test_score_negative_values() {
        let q = QuantizedTensor { data: vec![-100i8; 4], scale: 0.01 };
        let k = QuantizedTensor { data: vec![100i8; 4], scale: 0.01 };
        let mut scores = vec![0.0f32; 1];
        quantized_score_computation(&q, &k, 1, 4, 1.0, &mut scores).unwrap();
        assert!(scores[0] < 0.0);
    }

    #[test]
    fn test_pipeline_preserves_relative_magnitude() {
        let seq_len = 4;
        let head_dim = 16;
        let (w, ws) = identity_i8(head_dim);
        // Row 0 has larger magnitude than row 1.
        let mut input = vec![0.01f32; seq_len * head_dim];
        for d in 0..head_dim {
            input[d] = 0.5;
        }
        let cfg = QuantizedAttentionConfig::new(seq_len, head_dim, 1);
        let mut output = vec![0.0f32; seq_len * head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_i4_clamp_range() {
        let input = vec![100.0, -100.0, 0.0];
        let qt = quantize_to_i4(&input);
        let unpacked = unpack_i4_to_i8(&qt);
        assert_eq!(unpacked[0], 7);
        assert_eq!(unpacked[1], -7);
        assert_eq!(unpacked[2], 0);
    }

    #[test]
    fn test_cosine_similarity_helper() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0, 1e-6));

        let c = vec![1.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &c), 1.0, 1e-6));
    }

    #[test]
    fn test_avx2_detection_does_not_panic() {
        // Should never panic regardless of hardware.
        let _ = has_avx2();
    }

    #[test]
    fn test_full_pipeline_seq8_dim32() {
        let seq_len = 8;
        let head_dim = 32;
        let (w, ws) = identity_i8(head_dim);
        let input: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let cfg = QuantizedAttentionConfig::new(seq_len, head_dim, 1);
        let mut output = vec![0.0f32; seq_len * head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        // All finite and non-trivial.
        assert!(output.iter().all(|v| v.is_finite()));
        let norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.01);
    }

    #[test]
    fn test_full_pipeline_causal_seq8() {
        let seq_len = 8;
        let head_dim = 16;
        let (w, ws) = identity_i8(head_dim);
        let input: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i as f32) * 0.13).cos() * 0.3).collect();
        let mut cfg = QuantizedAttentionConfig::new(seq_len, head_dim, 1);
        cfg.causal = true;
        let mut output = vec![0.0f32; seq_len * head_dim];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_score_computation_symmetry() {
        let seq = 3;
        let hd = 8;
        let data: Vec<i8> = (0..seq * hd).map(|i| (i % 127) as i8).collect();
        let q = QuantizedTensor { data: data.clone(), scale: 0.01 };
        let k = QuantizedTensor { data, scale: 0.01 };
        let mut scores = vec![0.0f32; seq * seq];
        quantized_score_computation(&q, &k, seq, hd, 1.0, &mut scores).unwrap();
        // Q == K → scores should be symmetric.
        for i in 0..seq {
            for j in 0..seq {
                assert!(
                    approx_eq(scores[i * seq + j], scores[j * seq + i], 1e-6),
                    "scores[{i},{j}] != scores[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_quantize_i8_large_vector() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let qt = quantize_to_i8(&input);
        assert_eq!(qt.data.len(), n);
        let deq = dequantize_i8(&qt);
        let max_err: f32 =
            input.iter().zip(&deq).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 0.01);
    }

    #[test]
    fn test_softmax_idempotence_property() {
        let mut scores = vec![0.2, 0.3, 0.5, 0.1, 0.1, 0.8, 0.4, 0.2, 0.4];
        quantized_softmax(&mut scores, 3, false).unwrap();
        let first_pass = scores.clone();
        // Applying softmax again changes values.
        quantized_softmax(&mut scores, 3, false).unwrap();
        // But rows still sum to 1.
        for row in 0..3 {
            let sum: f32 = scores[row * 3..(row + 1) * 3].iter().sum();
            assert!(approx_eq(sum, 1.0, 1e-5));
        }
        // And values should be different from first pass (softmax is not idempotent).
        let changed = first_pass.iter().zip(&scores).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "softmax should not be idempotent on non-uniform input");
    }

    #[test]
    fn test_projection_scale_propagation() {
        let dim = 4;
        let weight = vec![1i8; dim * dim];
        let input = vec![1.0f32; dim];
        let mut out_s1 = vec![0.0f32; dim];
        let mut out_s2 = vec![0.0f32; dim];
        quantized_qkv_projection(&input, &weight, 1.0, dim, dim, &mut out_s1).unwrap();
        quantized_qkv_projection(&input, &weight, 2.0, dim, dim, &mut out_s2).unwrap();
        // Scale 2 should give double the output.
        for (a, b) in out_s1.iter().zip(&out_s2) {
            assert!(approx_eq(*b, *a * 2.0, 1e-4));
        }
    }

    #[test]
    fn test_dequantized_output_bias_additivity() {
        let q = vec![0i8; 4];
        let bias = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        dequantized_output(&q, 1.0, Some(&bias), &mut out).unwrap();
        // With zero quantized values, output = bias.
        assert!(vec_approx_eq(&out, &bias, 1e-6));
    }

    #[test]
    fn test_i4_round_trip_many_values() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let qt = quantize_to_i4(&input);
        let unpacked = unpack_i4_to_i8(&qt);
        assert_eq!(unpacked.len(), 32);
        // Check that signs are preserved for non-zero values.
        for (i, &v) in input.iter().enumerate() {
            if v.abs() > qt.scale * 0.5 {
                let same_sign = (v > 0.0 && unpacked[i] > 0) || (v < 0.0 && unpacked[i] < 0);
                assert!(same_sign, "sign mismatch at index {i}");
            }
        }
    }

    #[test]
    fn test_value_aggregation_one_hot_weights() {
        let seq = 3;
        let hd = 2;
        // Weights: each row selects a specific V row.
        let weights = vec![
            0.0, 1.0, 0.0, // row 0 selects V[1]
            0.0, 0.0, 1.0, // row 1 selects V[2]
            1.0, 0.0, 0.0, // row 2 selects V[0]
        ];
        let v = QuantizedTensor { data: vec![10, 20, 30, 40, 50, 60], scale: 1.0 };
        let mut output = vec![0.0f32; seq * hd];
        quantized_value_aggregation(&weights, &v, seq, hd, &mut output).unwrap();
        // Row 0 → V[1] = [30, 40].
        assert!(approx_eq(output[0], 30.0, 1e-3));
        assert!(approx_eq(output[1], 40.0, 1e-3));
        // Row 1 → V[2] = [50, 60].
        assert!(approx_eq(output[2], 50.0, 1e-3));
        assert!(approx_eq(output[3], 60.0, 1e-3));
        // Row 2 → V[0] = [10, 20].
        assert!(approx_eq(output[4], 10.0, 1e-3));
        assert!(approx_eq(output[5], 20.0, 1e-3));
    }

    #[test]
    fn test_score_computation_zero_scale() {
        let q = QuantizedTensor { data: vec![100i8; 4], scale: 0.0 };
        let k = QuantizedTensor { data: vec![100i8; 4], scale: 1.0 };
        let mut scores = vec![99.0f32; 1];
        quantized_score_computation(&q, &k, 1, 4, 1.0, &mut scores).unwrap();
        assert!(approx_eq(scores[0], 0.0, 1e-6));
    }

    #[test]
    fn test_self_attention_all_zeros() {
        let seq = 2;
        let hd = 4;
        let (w, ws) = identity_i8(hd);
        let input = vec![0.0f32; seq * hd];
        let cfg = QuantizedAttentionConfig::new(seq, hd, 1);
        let mut output = vec![99.0f32; seq * hd];
        quantized_self_attention(&input, &w, ws, &w, ws, &w, ws, &cfg, &mut output).unwrap();
        assert!(output.iter().all(|&v| approx_eq(v, 0.0, 1e-6)));
    }

    #[test]
    fn test_quantize_i8_negative_only() {
        let input = vec![-1.0, -0.5, -0.25];
        let qt = quantize_to_i8(&input);
        assert!(qt.data.iter().all(|&v| v <= 0));
    }

    #[test]
    fn test_quantize_i8_positive_only() {
        let input = vec![1.0, 0.5, 0.25];
        let qt = quantize_to_i8(&input);
        assert!(qt.data.iter().all(|&v| v >= 0));
    }

    #[test]
    fn test_softmax_causal_two_tokens() {
        let mut scores = vec![2.0, 3.0, 4.0, 5.0];
        quantized_softmax(&mut scores, 2, true).unwrap();
        // Row 0: only position 0 visible.
        assert!(approx_eq(scores[0], 1.0, 1e-5));
        assert!(approx_eq(scores[1], 0.0, 1e-5));
        // Row 1: both visible.
        let sum: f32 = scores[2..4].iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));
    }

    #[test]
    fn test_dequantized_output_preserves_zero() {
        let q = vec![0i8; 8];
        let mut out = vec![99.0f32; 8];
        dequantized_output(&q, 0.5, None, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }
}
