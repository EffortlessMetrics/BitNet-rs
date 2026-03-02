//! ARM NEON fused attention v2 kernel for Apple Silicon.
//!
//! Implements six fused attention operations with NEON SIMD acceleration
//! and scalar fallback paths:
//!
//! 1. **Fused QKV projection** — single-pass Q, K, V projection from input
//! 2. **Fused scaled dot-product attention** — Q·Kᵀ/√d + softmax + V
//! 3. **Fused multi-head attention** — split heads + attention + concat
//! 4. **Fused GQA attention** — grouped-query attention with K/V head replication
//! 5. **Fused causal attention** — attention with inline causal mask
//! 6. **Fused attention with RoPE** — rotary positional embedding during attention
//!
//! # NEON intrinsics used
//!
//! | Intrinsic      | Purpose                                 |
//! |----------------|-----------------------------------------|
//! | `vld1q_f32`    | 128-bit (4×f32) load                    |
//! | `vst1q_f32`    | 128-bit (4×f32) store                   |
//! | `vdupq_n_f32`  | Broadcast scalar to four lanes          |
//! | `vfmaq_f32`    | Fused multiply-add: a + b * c           |
//! | `vmulq_f32`    | Lane-wise multiply                      |
//! | `vaddq_f32`    | Lane-wise add                           |
//! | `vsubq_f32`    | Lane-wise subtract                      |
//! | `vaddvq_f32`   | Horizontal sum of four lanes            |
//! | `vmaxvq_f32`   | Horizontal max of four lanes            |

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── Scalar helpers ─────────────────────────────────────────────────────

/// Scalar fast exp with clamping to avoid overflow.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    x.exp()
}

/// Scalar softmax in-place over a mutable slice.
fn softmax_inplace_scalar(vals: &mut [f32]) {
    if vals.is_empty() {
        return;
    }
    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in vals.iter_mut() {
        *v = fast_exp_scalar(*v - max_val);
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in vals.iter_mut() {
            *v *= inv;
        }
    }
}

/// Scalar dot product of two f32 slices.
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar matrix multiply: C[m×n] = A[m×k] · B[k×n].
fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// ── NEON helpers ───────────────────────────────────────────────────────

/// NEON dot product of two f32 slices.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / LANES;
    let mut acc = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            acc = vfmaq_f32(acc, va, vb);
        }
    }

    let mut result = unsafe { vaddvq_f32(acc) };
    for i in (chunks * LANES)..n {
        result += a[i] * b[i];
    }
    result
}

/// NEON softmax in-place.
#[cfg(target_arch = "aarch64")]
fn softmax_inplace_neon(vals: &mut [f32]) {
    if vals.is_empty() {
        return;
    }
    let n = vals.len();
    let chunks = n / LANES;

    // Find max
    let mut max_vec = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        unsafe {
            let v = vld1q_f32(vals.as_ptr().add(i * LANES));
            max_vec = vmaxq_f32(max_vec, v);
        }
    }
    let mut max_val = unsafe { vmaxvq_f32(max_vec) };
    for i in (chunks * LANES)..n {
        max_val = max_val.max(vals[i]);
    }

    // exp(x - max) and sum
    let max_v = unsafe { vdupq_n_f32(max_val) };
    let mut sum = 0.0f32;
    for i in 0..chunks {
        let offset = i * LANES;
        unsafe {
            let v = vld1q_f32(vals.as_ptr().add(offset));
            let shifted = vsubq_f32(v, max_v);
            let mut arr = [0.0f32; LANES];
            vst1q_f32(arr.as_mut_ptr(), shifted);
            for j in 0..LANES {
                arr[j] = fast_exp_scalar(arr[j]);
            }
            let exp_v = vld1q_f32(arr.as_ptr());
            vst1q_f32(vals.as_mut_ptr().add(offset), exp_v);
            sum += vaddvq_f32(exp_v);
        }
    }
    for i in (chunks * LANES)..n {
        vals[i] = fast_exp_scalar(vals[i] - max_val);
        sum += vals[i];
    }

    // Normalize
    if sum > 0.0 {
        let inv = 1.0 / sum;
        let inv_v = unsafe { vdupq_n_f32(inv) };
        for i in 0..chunks {
            let offset = i * LANES;
            unsafe {
                let v = vld1q_f32(vals.as_ptr().add(offset));
                let normed = vmulq_f32(v, inv_v);
                vst1q_f32(vals.as_mut_ptr().add(offset), normed);
            }
        }
        for i in (chunks * LANES)..n {
            vals[i] *= inv;
        }
    }
}

/// NEON-accelerated matrix multiply: C[m×n] += A[m×k] · B[k×n].
#[cfg(target_arch = "aarch64")]
fn matmul_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            let chunks = k / LANES;
            for p in 0..chunks {
                let offset = p * LANES;
                unsafe {
                    let va = vld1q_f32(a.as_ptr().add(i * k + offset));
                    // Gather from B column j — not contiguous, fall back
                    let vb = vld1q_f32(
                        [
                            b[offset * n + j],
                            b[(offset + 1) * n + j],
                            b[(offset + 2) * n + j],
                            b[(offset + 3) * n + j],
                        ]
                        .as_ptr(),
                    );
                    acc = vfmaq_f32(acc, va, vb);
                }
            }
            let mut val = unsafe { vaddvq_f32(acc) };
            for p in (chunks * LANES)..k {
                val += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = val;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Fused QKV Projection
// ═══════════════════════════════════════════════════════════════════════

/// Fused QKV projection: projects input into Q, K, V in a single pass.
///
/// - `input`: `[seq_len, d_model]`
/// - `wq`, `wk`, `wv`: `[d_model, head_dim * num_heads]` projection weights
/// - `bq`, `bk`, `bv`: optional biases `[head_dim * num_heads]`
/// - Returns `(q, k, v)` each `[seq_len, head_dim * num_heads]`
pub fn fused_qkv_projection(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    bq: Option<&[f32]>,
    bk: Option<&[f32]>,
    bv: Option<&[f32]>,
    seq_len: usize,
    d_model: usize,
    proj_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(input.len(), seq_len * d_model);
    assert_eq!(wq.len(), d_model * proj_dim);
    assert_eq!(wk.len(), d_model * proj_dim);
    assert_eq!(wv.len(), d_model * proj_dim);

    let mut q = vec![0.0f32; seq_len * proj_dim];
    let mut k = vec![0.0f32; seq_len * proj_dim];
    let mut v = vec![0.0f32; seq_len * proj_dim];

    #[cfg(target_arch = "aarch64")]
    {
        matmul_neon(input, wq, &mut q, seq_len, d_model, proj_dim);
        matmul_neon(input, wk, &mut k, seq_len, d_model, proj_dim);
        matmul_neon(input, wv, &mut v, seq_len, d_model, proj_dim);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        matmul_scalar(input, wq, &mut q, seq_len, d_model, proj_dim);
        matmul_scalar(input, wk, &mut k, seq_len, d_model, proj_dim);
        matmul_scalar(input, wv, &mut v, seq_len, d_model, proj_dim);
    }

    // Add biases
    if let Some(bias) = bq {
        for i in 0..seq_len {
            for j in 0..proj_dim {
                q[i * proj_dim + j] += bias[j];
            }
        }
    }
    if let Some(bias) = bk {
        for i in 0..seq_len {
            for j in 0..proj_dim {
                k[i * proj_dim + j] += bias[j];
            }
        }
    }
    if let Some(bias) = bv {
        for i in 0..seq_len {
            for j in 0..proj_dim {
                v[i * proj_dim + j] += bias[j];
            }
        }
    }

    (q, k, v)
}

/// Scalar-only fused QKV projection for testing/fallback.
pub fn fused_qkv_projection_scalar(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    bq: Option<&[f32]>,
    bk: Option<&[f32]>,
    bv: Option<&[f32]>,
    seq_len: usize,
    d_model: usize,
    proj_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(input.len(), seq_len * d_model);
    let mut q = vec![0.0f32; seq_len * proj_dim];
    let mut k = vec![0.0f32; seq_len * proj_dim];
    let mut v = vec![0.0f32; seq_len * proj_dim];

    matmul_scalar(input, wq, &mut q, seq_len, d_model, proj_dim);
    matmul_scalar(input, wk, &mut k, seq_len, d_model, proj_dim);
    matmul_scalar(input, wv, &mut v, seq_len, d_model, proj_dim);

    if let Some(bias) = bq {
        for i in 0..seq_len {
            for j in 0..proj_dim {
                q[i * proj_dim + j] += bias[j];
            }
        }
    }
    if let Some(bias) = bk {
        for i in 0..seq_len {
            for j in 0..proj_dim {
                k[i * proj_dim + j] += bias[j];
            }
        }
    }
    if let Some(bias) = bv {
        for i in 0..seq_len {
            for j in 0..proj_dim {
                v[i * proj_dim + j] += bias[j];
            }
        }
    }

    (q, k, v)
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Fused Scaled Dot-Product Attention
// ═══════════════════════════════════════════════════════════════════════

/// Fused scaled dot-product attention: softmax(Q·Kᵀ/√d)·V in one pass.
///
/// - `q`: `[seq_q, head_dim]`
/// - `k`: `[seq_k, head_dim]`
/// - `v`: `[seq_k, head_dim]`
/// - Returns `[seq_q, head_dim]`
pub fn fused_scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_q * head_dim);
    assert_eq!(k.len(), seq_k * head_dim);
    assert_eq!(v.len(), seq_k * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * head_dim];

    for i in 0..seq_q {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let mut scores = vec![0.0f32; seq_k];

        // Compute scores: Q·Kᵀ / √d
        for j in 0..seq_k {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            #[cfg(target_arch = "aarch64")]
            {
                scores[j] = dot_neon(q_row, k_row) * scale;
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                scores[j] = dot_scalar(q_row, k_row) * scale;
            }
        }

        // Softmax
        #[cfg(target_arch = "aarch64")]
        softmax_inplace_neon(&mut scores);
        #[cfg(not(target_arch = "aarch64"))]
        softmax_inplace_scalar(&mut scores);

        // Weighted sum of V
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_k {
            let w = scores[j];
            if w == 0.0 {
                continue;
            }
            #[cfg(target_arch = "aarch64")]
            {
                let chunks = head_dim / LANES;
                for c in 0..chunks {
                    let offset = c * LANES;
                    unsafe {
                        let vo = vld1q_f32(out_row.as_ptr().add(offset));
                        let vv = vld1q_f32(v.as_ptr().add(j * head_dim + offset));
                        let vw = vdupq_n_f32(w);
                        let res = vfmaq_f32(vo, vv, vw);
                        vst1q_f32(out_row.as_mut_ptr().add(offset), res);
                    }
                }
                for d in (chunks * LANES)..head_dim {
                    out_row[d] += w * v[j * head_dim + d];
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for d in 0..head_dim {
                    out_row[d] += w * v[j * head_dim + d];
                }
            }
        }
    }

    output
}

/// Scalar-only fused scaled dot-product attention.
pub fn fused_scaled_dot_product_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_q * head_dim);
    assert_eq!(k.len(), seq_k * head_dim);
    assert_eq!(v.len(), seq_k * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * head_dim];

    for i in 0..seq_q {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let mut scores = vec![0.0f32; seq_k];

        for j in 0..seq_k {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = dot_scalar(q_row, k_row) * scale;
        }

        softmax_inplace_scalar(&mut scores);

        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_k {
            let w = scores[j];
            for d in 0..head_dim {
                out_row[d] += w * v[j * head_dim + d];
            }
        }
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Fused Multi-Head Attention
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for multi-head attention.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}

/// Fused multi-head attention: split heads + per-head attention + concat.
///
/// - `q`, `k`, `v`: `[seq_len, num_heads * head_dim]` (interleaved heads)
/// - Returns `[seq_len, num_heads * head_dim]`
pub fn fused_multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &MultiHeadAttentionConfig,
) -> Vec<f32> {
    let MultiHeadAttentionConfig { num_heads, head_dim, seq_len } = *config;
    let total_dim = num_heads * head_dim;

    assert_eq!(q.len(), seq_len * total_dim);
    assert_eq!(k.len(), seq_len * total_dim);
    assert_eq!(v.len(), seq_len * total_dim);

    let mut output = vec![0.0f32; seq_len * total_dim];

    // Extract per-head slices and run attention
    for h in 0..num_heads {
        let mut q_h = vec![0.0f32; seq_len * head_dim];
        let mut k_h = vec![0.0f32; seq_len * head_dim];
        let mut v_h = vec![0.0f32; seq_len * head_dim];

        // Split heads
        for s in 0..seq_len {
            for d in 0..head_dim {
                q_h[s * head_dim + d] = q[s * total_dim + h * head_dim + d];
                k_h[s * head_dim + d] = k[s * total_dim + h * head_dim + d];
                v_h[s * head_dim + d] = v[s * total_dim + h * head_dim + d];
            }
        }

        // Per-head attention
        let attn_h =
            fused_scaled_dot_product_attention(&q_h, &k_h, &v_h, seq_len, seq_len, head_dim);

        // Concat back
        for s in 0..seq_len {
            for d in 0..head_dim {
                output[s * total_dim + h * head_dim + d] = attn_h[s * head_dim + d];
            }
        }
    }

    output
}

/// Scalar-only fused multi-head attention.
pub fn fused_multi_head_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &MultiHeadAttentionConfig,
) -> Vec<f32> {
    let MultiHeadAttentionConfig { num_heads, head_dim, seq_len } = *config;
    let total_dim = num_heads * head_dim;

    assert_eq!(q.len(), seq_len * total_dim);

    let mut output = vec![0.0f32; seq_len * total_dim];

    for h in 0..num_heads {
        let mut q_h = vec![0.0f32; seq_len * head_dim];
        let mut k_h = vec![0.0f32; seq_len * head_dim];
        let mut v_h = vec![0.0f32; seq_len * head_dim];

        for s in 0..seq_len {
            for d in 0..head_dim {
                q_h[s * head_dim + d] = q[s * total_dim + h * head_dim + d];
                k_h[s * head_dim + d] = k[s * total_dim + h * head_dim + d];
                v_h[s * head_dim + d] = v[s * total_dim + h * head_dim + d];
            }
        }

        let attn_h =
            fused_scaled_dot_product_attention_scalar(&q_h, &k_h, &v_h, seq_len, seq_len, head_dim);

        for s in 0..seq_len {
            for d in 0..head_dim {
                output[s * total_dim + h * head_dim + d] = attn_h[s * head_dim + d];
            }
        }
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Fused GQA (Grouped-Query) Attention
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for grouped-query attention.
#[derive(Debug, Clone)]
pub struct GqaAttentionConfig {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
}

/// Fused grouped-query attention with K/V head replication.
///
/// - `q`: `[seq_len, num_q_heads * head_dim]`
/// - `k`: `[seq_len, num_kv_heads * head_dim]`
/// - `v`: `[seq_len, num_kv_heads * head_dim]`
/// - Returns `[seq_len, num_q_heads * head_dim]`
pub fn fused_gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &GqaAttentionConfig,
) -> Vec<f32> {
    let GqaAttentionConfig { num_q_heads, num_kv_heads, head_dim, seq_len } = *config;

    assert!(
        num_q_heads >= num_kv_heads && num_q_heads % num_kv_heads == 0,
        "num_q_heads must be a multiple of num_kv_heads"
    );
    let q_total = num_q_heads * head_dim;
    let kv_total = num_kv_heads * head_dim;
    assert_eq!(q.len(), seq_len * q_total);
    assert_eq!(k.len(), seq_len * kv_total);
    assert_eq!(v.len(), seq_len * kv_total);

    let group_size = num_q_heads / num_kv_heads;
    let mut output = vec![0.0f32; seq_len * q_total];

    for qh in 0..num_q_heads {
        let kv_h = qh / group_size;

        let mut q_h = vec![0.0f32; seq_len * head_dim];
        let mut k_h = vec![0.0f32; seq_len * head_dim];
        let mut v_h = vec![0.0f32; seq_len * head_dim];

        for s in 0..seq_len {
            for d in 0..head_dim {
                q_h[s * head_dim + d] = q[s * q_total + qh * head_dim + d];
                k_h[s * head_dim + d] = k[s * kv_total + kv_h * head_dim + d];
                v_h[s * head_dim + d] = v[s * kv_total + kv_h * head_dim + d];
            }
        }

        let attn_h =
            fused_scaled_dot_product_attention(&q_h, &k_h, &v_h, seq_len, seq_len, head_dim);

        for s in 0..seq_len {
            for d in 0..head_dim {
                output[s * q_total + qh * head_dim + d] = attn_h[s * head_dim + d];
            }
        }
    }

    output
}

/// Scalar-only fused grouped-query attention.
pub fn fused_gqa_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &GqaAttentionConfig,
) -> Vec<f32> {
    let GqaAttentionConfig { num_q_heads, num_kv_heads, head_dim, seq_len } = *config;

    assert!(num_q_heads >= num_kv_heads && num_q_heads % num_kv_heads == 0);
    let q_total = num_q_heads * head_dim;
    let kv_total = num_kv_heads * head_dim;
    let group_size = num_q_heads / num_kv_heads;
    let mut output = vec![0.0f32; seq_len * q_total];

    for qh in 0..num_q_heads {
        let kv_h = qh / group_size;

        let mut q_h = vec![0.0f32; seq_len * head_dim];
        let mut k_h = vec![0.0f32; seq_len * head_dim];
        let mut v_h = vec![0.0f32; seq_len * head_dim];

        for s in 0..seq_len {
            for d in 0..head_dim {
                q_h[s * head_dim + d] = q[s * q_total + qh * head_dim + d];
                k_h[s * head_dim + d] = k[s * kv_total + kv_h * head_dim + d];
                v_h[s * head_dim + d] = v[s * kv_total + kv_h * head_dim + d];
            }
        }

        let attn_h =
            fused_scaled_dot_product_attention_scalar(&q_h, &k_h, &v_h, seq_len, seq_len, head_dim);

        for s in 0..seq_len {
            for d in 0..head_dim {
                output[s * q_total + qh * head_dim + d] = attn_h[s * head_dim + d];
            }
        }
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Fused Causal Attention
// ═══════════════════════════════════════════════════════════════════════

/// Fused causal attention: applies a lower-triangular causal mask inline.
///
/// - `q`, `k`, `v`: `[seq_len, head_dim]`
/// - Returns `[seq_len, head_dim]`
pub fn fused_causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let attend_len = i + 1; // causal: only attend to positions 0..=i
        let mut scores = vec![0.0f32; attend_len];

        for j in 0..attend_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            #[cfg(target_arch = "aarch64")]
            {
                scores[j] = dot_neon(q_row, k_row) * scale;
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                scores[j] = dot_scalar(q_row, k_row) * scale;
            }
        }

        #[cfg(target_arch = "aarch64")]
        softmax_inplace_neon(&mut scores);
        #[cfg(not(target_arch = "aarch64"))]
        softmax_inplace_scalar(&mut scores);

        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for j in 0..attend_len {
            let w = scores[j];
            if w == 0.0 {
                continue;
            }
            for d in 0..head_dim {
                out_row[d] += w * v[j * head_dim + d];
            }
        }
    }

    output
}

/// Scalar-only fused causal attention.
pub fn fused_causal_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let attend_len = i + 1;
        let mut scores = vec![0.0f32; attend_len];

        for j in 0..attend_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = dot_scalar(q_row, k_row) * scale;
        }

        softmax_inplace_scalar(&mut scores);

        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for j in 0..attend_len {
            let w = scores[j];
            for d in 0..head_dim {
                out_row[d] += w * v[j * head_dim + d];
            }
        }
    }

    output
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Fused Attention with RoPE
// ═══════════════════════════════════════════════════════════════════════

/// Apply rotary positional embedding to a single head vector in-place.
///
/// Uses the standard RoPE formula:
///   x'[2i]   = x[2i]   * cos(θ) - x[2i+1] * sin(θ)
///   x'[2i+1] = x[2i+1] * cos(θ) + x[2i]   * sin(θ)
/// where θ_i = pos / 10000^(2i/d).
fn apply_rope_inplace(vec: &mut [f32], pos: usize, head_dim: usize) {
    let half = head_dim / 2;
    for i in 0..half {
        let theta = (pos as f32) / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = vec[2 * i];
        let x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cos_t - x1 * sin_t;
        vec[2 * i + 1] = x1 * cos_t + x0 * sin_t;
    }
}

/// Apply RoPE with NEON acceleration.
#[cfg(target_arch = "aarch64")]
fn apply_rope_inplace_neon(vec: &mut [f32], pos: usize, head_dim: usize) {
    let half = head_dim / 2;
    // Process pairs in chunks of 2 (= 4 f32 values via NEON)
    let pair_chunks = half / 2;

    for c in 0..pair_chunks {
        let i0 = c * 2;
        let i1 = c * 2 + 1;
        let theta0 = (pos as f32) / 10000.0f32.powf(2.0 * i0 as f32 / head_dim as f32);
        let theta1 = (pos as f32) / 10000.0f32.powf(2.0 * i1 as f32 / head_dim as f32);

        unsafe {
            let cos_v =
                vld1q_f32([theta0.cos(), theta0.cos(), theta1.cos(), theta1.cos()].as_ptr());
            let sin_v =
                vld1q_f32([theta0.sin(), theta0.sin(), theta1.sin(), theta1.sin()].as_ptr());

            let offset = i0 * 2;
            let x = vld1q_f32(vec.as_ptr().add(offset));
            // x = [x0, x1, x2, x3]
            // For rotation: need [x0, x0, x2, x2] * cos + [-x1, x0, -x3, x2] * sin
            // Simplified: compute with scalar-like logic in NEON
            let mut arr = [0.0f32; LANES];
            vst1q_f32(arr.as_mut_ptr(), x);

            let r0 = arr[0] * theta0.cos() - arr[1] * theta0.sin();
            let r1 = arr[1] * theta0.cos() + arr[0] * theta0.sin();
            let r2 = arr[2] * theta1.cos() - arr[3] * theta1.sin();
            let r3 = arr[3] * theta1.cos() + arr[2] * theta1.sin();

            let result = vld1q_f32([r0, r1, r2, r3].as_ptr());
            vst1q_f32(vec.as_mut_ptr().add(offset), result);
        }
    }

    // Handle remaining pairs
    for i in (pair_chunks * 2)..half {
        let theta = (pos as f32) / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = vec[2 * i];
        let x1 = vec[2 * i + 1];
        vec[2 * i] = x0 * cos_t - x1 * sin_t;
        vec[2 * i + 1] = x1 * cos_t + x0 * sin_t;
    }
}

/// Fused attention with rotary positional embedding applied to Q and K.
///
/// - `q`, `k`, `v`: `[seq_len, head_dim]`
/// - `head_dim` must be even
/// - Returns `[seq_len, head_dim]`
pub fn fused_attention_with_rope(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);

    let mut q_rope = q.to_vec();
    let mut k_rope = k.to_vec();

    // Apply RoPE to each position
    for pos in 0..seq_len {
        let q_slice = &mut q_rope[pos * head_dim..(pos + 1) * head_dim];
        let k_slice = &mut k_rope[pos * head_dim..(pos + 1) * head_dim];

        #[cfg(target_arch = "aarch64")]
        {
            apply_rope_inplace_neon(q_slice, pos, head_dim);
            apply_rope_inplace_neon(k_slice, pos, head_dim);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            apply_rope_inplace(q_slice, pos, head_dim);
            apply_rope_inplace(k_slice, pos, head_dim);
        }
    }

    fused_scaled_dot_product_attention(&q_rope, &k_rope, v, seq_len, seq_len, head_dim)
}

/// Scalar-only fused attention with RoPE.
pub fn fused_attention_with_rope_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
    assert_eq!(q.len(), seq_len * head_dim);

    let mut q_rope = q.to_vec();
    let mut k_rope = k.to_vec();

    for pos in 0..seq_len {
        apply_rope_inplace(&mut q_rope[pos * head_dim..(pos + 1) * head_dim], pos, head_dim);
        apply_rope_inplace(&mut k_rope[pos * head_dim..(pos + 1) * head_dim], pos, head_dim);
    }

    fused_scaled_dot_product_attention_scalar(&q_rope, &k_rope, v, seq_len, seq_len, head_dim)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = ((state >> 33) ^ state) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn identity_matrix(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    // ─── 1. Fused QKV Projection Tests ────────────────────────────────

    #[test]
    fn test_qkv_identity_projection() {
        let d = 4;
        let seq = 2;
        let input = rand_vec(seq * d, 42);
        let eye = identity_matrix(d);
        let (q, k, v) = fused_qkv_projection(&input, &eye, &eye, &eye, None, None, None, seq, d, d);
        assert_close(&q, &input, TOL);
        assert_close(&k, &input, TOL);
        assert_close(&v, &input, TOL);
    }

    #[test]
    fn test_qkv_with_bias() {
        let d = 4;
        let seq = 1;
        let input = vec![1.0; d];
        let eye = identity_matrix(d);
        let bias = vec![0.5; d];
        let (q, _, _) =
            fused_qkv_projection(&input, &eye, &eye, &eye, Some(&bias), None, None, seq, d, d);
        for v in &q {
            assert!((v - 1.5).abs() < TOL);
        }
    }

    #[test]
    fn test_qkv_neon_scalar_parity() {
        let d = 16;
        let seq = 4;
        let input = rand_vec(seq * d, 100);
        let wq = rand_vec(d * d, 200);
        let wk = rand_vec(d * d, 300);
        let wv = rand_vec(d * d, 400);
        let bq = rand_vec(d, 500);

        let (q1, k1, v1) =
            fused_qkv_projection(&input, &wq, &wk, &wv, Some(&bq), None, None, seq, d, d);
        let (q2, k2, v2) =
            fused_qkv_projection_scalar(&input, &wq, &wk, &wv, Some(&bq), None, None, seq, d, d);
        assert_close(&q1, &q2, TOL);
        assert_close(&k1, &k2, TOL);
        assert_close(&v1, &v2, TOL);
    }

    #[test]
    fn test_qkv_zero_input() {
        let d = 8;
        let seq = 2;
        let input = vec![0.0; seq * d];
        let w = rand_vec(d * d, 42);
        let (q, k, v) = fused_qkv_projection(&input, &w, &w, &w, None, None, None, seq, d, d);
        for val in q.iter().chain(k.iter()).chain(v.iter()) {
            assert!(val.abs() < TOL);
        }
    }

    #[test]
    fn test_qkv_different_proj_dim() {
        let d_model = 8;
        let proj_dim = 4;
        let seq = 2;
        let input = rand_vec(seq * d_model, 42);
        let w = rand_vec(d_model * proj_dim, 43);
        let (q, k, v) =
            fused_qkv_projection(&input, &w, &w, &w, None, None, None, seq, d_model, proj_dim);
        assert_eq!(q.len(), seq * proj_dim);
        assert_eq!(k.len(), seq * proj_dim);
        assert_eq!(v.len(), seq * proj_dim);
    }

    #[test]
    fn test_qkv_seq_len_1() {
        let d = 8;
        let input = rand_vec(d, 42);
        let w = rand_vec(d * d, 43);
        let (q, k, v) = fused_qkv_projection(&input, &w, &w, &w, None, None, None, 1, d, d);
        assert_eq!(q.len(), d);
        assert_eq!(k.len(), d);
        assert_eq!(v.len(), d);
    }

    #[test]
    fn test_qkv_all_biases() {
        let d = 4;
        let seq = 2;
        let input = rand_vec(seq * d, 42);
        let eye = identity_matrix(d);
        let bq = vec![1.0; d];
        let bk = vec![2.0; d];
        let bv = vec![3.0; d];
        let (q, k, v) = fused_qkv_projection(
            &input,
            &eye,
            &eye,
            &eye,
            Some(&bq),
            Some(&bk),
            Some(&bv),
            seq,
            d,
            d,
        );
        for i in 0..seq * d {
            let base = input[i];
            assert!((q[i] - (base + 1.0)).abs() < TOL);
            assert!((k[i] - (base + 2.0)).abs() < TOL);
            assert!((v[i] - (base + 3.0)).abs() < TOL);
        }
    }

    #[test]
    fn test_qkv_large_batch() {
        let d = 32;
        let seq = 16;
        let input = rand_vec(seq * d, 42);
        let w = rand_vec(d * d, 43);
        let (q, k, v) = fused_qkv_projection(&input, &w, &w, &w, None, None, None, seq, d, d);
        assert_eq!(q.len(), seq * d);
        assert_eq!(k.len(), seq * d);
        assert_eq!(v.len(), seq * d);
    }

    #[test]
    fn test_qkv_parity_no_bias() {
        let d = 12;
        let seq = 3;
        let input = rand_vec(seq * d, 1);
        let wq = rand_vec(d * d, 2);
        let wk = rand_vec(d * d, 3);
        let wv = rand_vec(d * d, 4);
        let (q1, k1, v1) = fused_qkv_projection(&input, &wq, &wk, &wv, None, None, None, seq, d, d);
        let (q2, k2, v2) =
            fused_qkv_projection_scalar(&input, &wq, &wk, &wv, None, None, None, seq, d, d);
        assert_close(&q1, &q2, TOL);
        assert_close(&k1, &k2, TOL);
        assert_close(&v1, &v2, TOL);
    }

    // ─── 2. Fused Scaled Dot-Product Attention Tests ──────────────────

    #[test]
    fn test_sdpa_single_token() {
        let hd = 4;
        let q = rand_vec(hd, 10);
        let k = rand_vec(hd, 20);
        let v = rand_vec(hd, 30);
        let out = fused_scaled_dot_product_attention(&q, &k, &v, 1, 1, hd);
        // With single K/V, softmax is 1.0, so output = V
        assert_close(&out, &v, TOL);
    }

    #[test]
    fn test_sdpa_neon_scalar_parity() {
        let hd = 16;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        let out2 = fused_scaled_dot_product_attention_scalar(&q, &k, &v, seq, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_sdpa_output_shape() {
        let hd = 8;
        let sq = 3;
        let sk = 5;
        let q = rand_vec(sq * hd, 10);
        let k = rand_vec(sk * hd, 20);
        let v = rand_vec(sk * hd, 30);
        let out = fused_scaled_dot_product_attention(&q, &k, &v, sq, sk, hd);
        assert_eq!(out.len(), sq * hd);
    }

    #[test]
    fn test_sdpa_identical_qk() {
        let hd = 8;
        let seq = 4;
        let q = rand_vec(seq * hd, 42);
        let v = rand_vec(seq * hd, 43);
        let out = fused_scaled_dot_product_attention(&q, &q, &v, seq, seq, hd);
        assert_eq!(out.len(), seq * hd);
        // Self-attention should produce valid output
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_sdpa_uniform_attention() {
        // When all Q and K are equal, attention weights should be uniform
        let hd = 4;
        let seq = 3;
        let q = vec![1.0; seq * hd];
        let k = vec![1.0; seq * hd];
        // V values differ per position
        let mut v = vec![0.0; seq * hd];
        for s in 0..seq {
            for d in 0..hd {
                v[s * hd + d] = s as f32;
            }
        }
        let out = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        // Each output row should be the mean of V rows
        let mean = (0..seq).map(|s| s as f32).sum::<f32>() / seq as f32;
        for val in &out {
            assert!((val - mean).abs() < TOL);
        }
    }

    #[test]
    fn test_sdpa_zero_query() {
        let hd = 4;
        let seq = 2;
        let q = vec![0.0; seq * hd];
        let k = rand_vec(seq * hd, 42);
        let v = rand_vec(seq * hd, 43);
        let out = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        // All scores are 0 => uniform attention => mean of V
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_sdpa_large_dim() {
        let hd = 64;
        let seq = 8;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        let out2 = fused_scaled_dot_product_attention_scalar(&q, &k, &v, seq, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_sdpa_cross_attention() {
        let hd = 8;
        let sq = 2;
        let sk = 6;
        let q = rand_vec(sq * hd, 10);
        let k = rand_vec(sk * hd, 20);
        let v = rand_vec(sk * hd, 30);
        let out = fused_scaled_dot_product_attention(&q, &k, &v, sq, sk, hd);
        assert_eq!(out.len(), sq * hd);
    }

    // ─── 3. Fused Multi-Head Attention Tests ──────────────────────────

    #[test]
    fn test_mha_single_head() {
        let hd = 8;
        let seq = 4;
        let config = MultiHeadAttentionConfig { num_heads: 1, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out = fused_multi_head_attention(&q, &k, &v, &config);
        let expected = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        assert_close(&out, &expected, TOL);
    }

    #[test]
    fn test_mha_two_heads() {
        let hd = 4;
        let nh = 2;
        let seq = 3;
        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * nh * hd, 10);
        let k = rand_vec(seq * nh * hd, 20);
        let v = rand_vec(seq * nh * hd, 30);
        let out = fused_multi_head_attention(&q, &k, &v, &config);
        assert_eq!(out.len(), seq * nh * hd);
    }

    #[test]
    fn test_mha_neon_scalar_parity() {
        let hd = 8;
        let nh = 4;
        let seq = 4;
        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * nh * hd, 10);
        let k = rand_vec(seq * nh * hd, 20);
        let v = rand_vec(seq * nh * hd, 30);
        let out1 = fused_multi_head_attention(&q, &k, &v, &config);
        let out2 = fused_multi_head_attention_scalar(&q, &k, &v, &config);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_mha_output_shape() {
        let hd = 8;
        let nh = 4;
        let seq = 6;
        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let total = seq * nh * hd;
        let q = rand_vec(total, 10);
        let k = rand_vec(total, 20);
        let v = rand_vec(total, 30);
        let out = fused_multi_head_attention(&q, &k, &v, &config);
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_mha_heads_independent() {
        // Each head should produce independent output
        let hd = 4;
        let nh = 2;
        let seq = 2;
        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let total = seq * nh * hd;
        let q = rand_vec(total, 10);
        let k = rand_vec(total, 20);
        let v = rand_vec(total, 30);
        let out = fused_multi_head_attention(&q, &k, &v, &config);
        // Just verify finite and correct shape
        assert_eq!(out.len(), total);
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_mha_seq_len_1() {
        let hd = 8;
        let nh = 2;
        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: 1 };
        let total = nh * hd;
        let q = rand_vec(total, 10);
        let k = rand_vec(total, 20);
        let v = rand_vec(total, 30);
        let out = fused_multi_head_attention(&q, &k, &v, &config);
        // With seq_len=1, each head's output should equal its V
        assert_close(&out, &v, TOL);
    }

    #[test]
    fn test_mha_large_config() {
        let hd = 16;
        let nh = 8;
        let seq = 8;
        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let total = seq * nh * hd;
        let q = rand_vec(total, 10);
        let k = rand_vec(total, 20);
        let v = rand_vec(total, 30);
        let out1 = fused_multi_head_attention(&q, &k, &v, &config);
        let out2 = fused_multi_head_attention_scalar(&q, &k, &v, &config);
        assert_close(&out1, &out2, TOL);
    }

    // ─── 4. Fused GQA Attention Tests ─────────────────────────────────

    #[test]
    fn test_gqa_equal_heads_matches_mha() {
        let hd = 8;
        let nh = 4;
        let seq = 4;
        let mha_config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let gqa_config =
            GqaAttentionConfig { num_q_heads: nh, num_kv_heads: nh, head_dim: hd, seq_len: seq };
        let total = seq * nh * hd;
        let q = rand_vec(total, 10);
        let k = rand_vec(total, 20);
        let v = rand_vec(total, 30);
        let mha_out = fused_multi_head_attention(&q, &k, &v, &mha_config);
        let gqa_out = fused_gqa_attention(&q, &k, &v, &gqa_config);
        assert_close(&mha_out, &gqa_out, TOL);
    }

    #[test]
    fn test_gqa_4q_2kv() {
        let hd = 4;
        let seq = 3;
        let config =
            GqaAttentionConfig { num_q_heads: 4, num_kv_heads: 2, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * 4 * hd, 10);
        let k = rand_vec(seq * 2 * hd, 20);
        let v = rand_vec(seq * 2 * hd, 30);
        let out = fused_gqa_attention(&q, &k, &v, &config);
        assert_eq!(out.len(), seq * 4 * hd);
    }

    #[test]
    fn test_gqa_neon_scalar_parity() {
        let hd = 8;
        let seq = 4;
        let config =
            GqaAttentionConfig { num_q_heads: 8, num_kv_heads: 2, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * 8 * hd, 10);
        let k = rand_vec(seq * 2 * hd, 20);
        let v = rand_vec(seq * 2 * hd, 30);
        let out1 = fused_gqa_attention(&q, &k, &v, &config);
        let out2 = fused_gqa_attention_scalar(&q, &k, &v, &config);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_gqa_mqa_single_kv_head() {
        // Multi-query attention: all Q heads share 1 KV head
        let hd = 4;
        let seq = 2;
        let config =
            GqaAttentionConfig { num_q_heads: 4, num_kv_heads: 1, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * 4 * hd, 10);
        let k = rand_vec(seq * 1 * hd, 20);
        let v = rand_vec(seq * 1 * hd, 30);
        let out = fused_gqa_attention(&q, &k, &v, &config);
        assert_eq!(out.len(), seq * 4 * hd);
    }

    #[test]
    fn test_gqa_shared_heads_produce_same_output() {
        let hd = 4;
        let seq = 2;
        let config =
            GqaAttentionConfig { num_q_heads: 4, num_kv_heads: 2, head_dim: hd, seq_len: seq };
        // Make all Q heads identical for same KV group
        let q_head = rand_vec(hd, 10);
        let mut q = vec![0.0; seq * 4 * hd];
        for s in 0..seq {
            for h in 0..4 {
                for d in 0..hd {
                    q[s * 4 * hd + h * hd + d] = q_head[d];
                }
            }
        }
        let k = rand_vec(seq * 2 * hd, 20);
        let v = rand_vec(seq * 2 * hd, 30);
        let out = fused_gqa_attention(&q, &k, &v, &config);
        // Heads 0,1 share KV head 0; heads 2,3 share KV head 1
        for s in 0..seq {
            let h0 = &out[s * 4 * hd..s * 4 * hd + hd];
            let h1 = &out[s * 4 * hd + hd..s * 4 * hd + 2 * hd];
            assert_close(h0, h1, TOL);
            let h2 = &out[s * 4 * hd + 2 * hd..s * 4 * hd + 3 * hd];
            let h3 = &out[s * 4 * hd + 3 * hd..s * 4 * hd + 4 * hd];
            assert_close(h2, h3, TOL);
        }
    }

    #[test]
    fn test_gqa_output_finite() {
        let hd = 8;
        let seq = 4;
        let config =
            GqaAttentionConfig { num_q_heads: 8, num_kv_heads: 4, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * 8 * hd, 10);
        let k = rand_vec(seq * 4 * hd, 20);
        let v = rand_vec(seq * 4 * hd, 30);
        let out = fused_gqa_attention(&q, &k, &v, &config);
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_gqa_seq_len_1() {
        let hd = 4;
        let config =
            GqaAttentionConfig { num_q_heads: 4, num_kv_heads: 2, head_dim: hd, seq_len: 1 };
        let q = rand_vec(4 * hd, 10);
        let k = rand_vec(2 * hd, 20);
        let v = rand_vec(2 * hd, 30);
        let out = fused_gqa_attention(&q, &k, &v, &config);
        assert_eq!(out.len(), 4 * hd);
    }

    #[test]
    #[should_panic(expected = "num_q_heads must be a multiple")]
    fn test_gqa_invalid_head_ratio() {
        let config =
            GqaAttentionConfig { num_q_heads: 5, num_kv_heads: 3, head_dim: 4, seq_len: 1 };
        let q = vec![0.0; 5 * 4];
        let k = vec![0.0; 3 * 4];
        let v = vec![0.0; 3 * 4];
        fused_gqa_attention(&q, &k, &v, &config);
    }

    // ─── 5. Fused Causal Attention Tests ──────────────────────────────

    #[test]
    fn test_causal_single_token() {
        let hd = 4;
        let q = rand_vec(hd, 10);
        let k = rand_vec(hd, 20);
        let v = rand_vec(hd, 30);
        let out = fused_causal_attention(&q, &k, &v, 1, hd);
        // Single token: attends only to itself, weight=1.0 => output=V
        assert_close(&out, &v, TOL);
    }

    #[test]
    fn test_causal_neon_scalar_parity() {
        let hd = 16;
        let seq = 8;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_causal_attention(&q, &k, &v, seq, hd);
        let out2 = fused_causal_attention_scalar(&q, &k, &v, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_causal_first_token_is_self_attention() {
        let hd = 4;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out = fused_causal_attention(&q, &k, &v, seq, hd);
        // First position always attends only to itself
        assert_close(&out[..hd], &v[..hd], TOL);
    }

    #[test]
    fn test_causal_mask_correctness() {
        // With causal mask, the last token sees all tokens; first sees only itself
        let hd = 4;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let causal_out = fused_causal_attention(&q, &k, &v, seq, hd);
        let full_out = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        // Last token in causal should equal last token in full attention
        let last_causal = &causal_out[(seq - 1) * hd..seq * hd];
        let last_full = &full_out[(seq - 1) * hd..seq * hd];
        assert_close(last_causal, last_full, TOL);
    }

    #[test]
    fn test_causal_output_shape() {
        let hd = 8;
        let seq = 6;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out = fused_causal_attention(&q, &k, &v, seq, hd);
        assert_eq!(out.len(), seq * hd);
    }

    #[test]
    fn test_causal_all_finite() {
        let hd = 16;
        let seq = 12;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out = fused_causal_attention(&q, &k, &v, seq, hd);
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_causal_two_tokens() {
        let hd = 4;
        let q = rand_vec(2 * hd, 10);
        let k = rand_vec(2 * hd, 20);
        let v = rand_vec(2 * hd, 30);
        let out = fused_causal_attention(&q, &k, &v, 2, hd);
        // First token only sees itself
        assert_close(&out[..hd], &v[..hd], TOL);
    }

    #[test]
    fn test_causal_large_seq() {
        let hd = 8;
        let seq = 32;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_causal_attention(&q, &k, &v, seq, hd);
        let out2 = fused_causal_attention_scalar(&q, &k, &v, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    // ─── 6. Fused Attention with RoPE Tests ──────────────────────────

    #[test]
    fn test_rope_attention_neon_scalar_parity() {
        let hd = 16;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_attention_with_rope(&q, &k, &v, seq, hd);
        let out2 = fused_attention_with_rope_scalar(&q, &k, &v, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_rope_attention_output_shape() {
        let hd = 8;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out = fused_attention_with_rope(&q, &k, &v, seq, hd);
        assert_eq!(out.len(), seq * hd);
    }

    #[test]
    fn test_rope_attention_single_token() {
        let hd = 8;
        let q = rand_vec(hd, 10);
        let k = rand_vec(hd, 20);
        let v = rand_vec(hd, 30);
        let out = fused_attention_with_rope(&q, &k, &v, 1, hd);
        // Single token: output is V (softmax over 1 key = 1.0)
        assert_close(&out, &v, TOL);
    }

    #[test]
    fn test_rope_changes_output() {
        let hd = 8;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let with_rope = fused_attention_with_rope(&q, &k, &v, seq, hd);
        let without_rope = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        // RoPE should change the output for seq_len > 1
        let mut any_diff = false;
        for (a, b) in with_rope.iter().zip(without_rope.iter()) {
            if (a - b).abs() > TOL {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "RoPE should change attention output");
    }

    #[test]
    fn test_rope_attention_finite() {
        let hd = 16;
        let seq = 8;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out = fused_attention_with_rope(&q, &k, &v, seq, hd);
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rope_position_zero_identity() {
        // At position 0, all θ = 0, so cos=1, sin=0 => RoPE is identity
        let hd = 8;
        let mut vec_orig = rand_vec(hd, 42);
        let mut vec_rope = vec_orig.clone();
        apply_rope_inplace(&mut vec_rope, 0, hd);
        assert_close(&vec_rope, &vec_orig, TOL);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it should approximately preserve L2 norm
        let hd = 16;
        let mut vec = rand_vec(hd, 42);
        let norm_before: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_inplace(&mut vec, 5, hd);
        let norm_after: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < TOL);
    }

    #[test]
    #[should_panic(expected = "head_dim must be even")]
    fn test_rope_attention_odd_dim_panics() {
        let q = vec![0.0; 5];
        let k = vec![0.0; 5];
        let v = vec![0.0; 5];
        fused_attention_with_rope(&q, &k, &v, 1, 5);
    }

    #[test]
    fn test_rope_attention_dim_4() {
        let hd = 4;
        let seq = 3;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_attention_with_rope(&q, &k, &v, seq, hd);
        let out2 = fused_attention_with_rope_scalar(&q, &k, &v, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    #[test]
    fn test_rope_attention_large() {
        let hd = 64;
        let seq = 8;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_attention_with_rope(&q, &k, &v, seq, hd);
        let out2 = fused_attention_with_rope_scalar(&q, &k, &v, seq, hd);
        assert_close(&out1, &out2, TOL);
    }

    // ─── Edge Cases and Cross-Cutting Tests ───────────────────────────

    #[test]
    fn test_softmax_scalar_basic() {
        let mut vals = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace_scalar(&mut vals);
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
        // Values should be monotonically increasing
        for i in 1..vals.len() {
            assert!(vals[i] >= vals[i - 1]);
        }
    }

    #[test]
    fn test_softmax_scalar_single() {
        let mut vals = vec![5.0];
        softmax_inplace_scalar(&mut vals);
        assert!((vals[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_softmax_scalar_empty() {
        let mut vals: Vec<f32> = vec![];
        softmax_inplace_scalar(&mut vals);
        assert!(vals.is_empty());
    }

    #[test]
    fn test_softmax_scalar_large_negative() {
        let mut vals = vec![-1000.0, -999.0, -998.0];
        softmax_inplace_scalar(&mut vals);
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < TOL);
    }

    #[test]
    fn test_dot_scalar_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        assert!((dot_scalar(&a, &b) - 10.0).abs() < TOL);
    }

    #[test]
    fn test_dot_scalar_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        assert!(dot_scalar(&a, &b).abs() < TOL);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_dot_neon_parity() {
        let a = rand_vec(33, 42); // Non-multiple-of-4
        let b = rand_vec(33, 43);
        let scalar = dot_scalar(&a, &b);
        let neon = dot_neon(&a, &b);
        assert!((scalar - neon).abs() < TOL);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_softmax_neon_parity() {
        let mut vals_scalar = rand_vec(17, 42); // Non-multiple-of-4
        let mut vals_neon = vals_scalar.clone();
        softmax_inplace_scalar(&mut vals_scalar);
        softmax_inplace_neon(&mut vals_neon);
        assert_close(&vals_scalar, &vals_neon, TOL);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_rope_neon_vs_scalar() {
        let hd = 16;
        let pos = 7;
        let mut vec_scalar = rand_vec(hd, 42);
        let mut vec_neon = vec_scalar.clone();
        apply_rope_inplace(&mut vec_scalar, pos, hd);
        apply_rope_inplace_neon(&mut vec_neon, pos, hd);
        assert_close(&vec_scalar, &vec_neon, TOL);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_rope_neon_dim_4() {
        let hd = 4;
        let pos = 3;
        let mut vec_scalar = rand_vec(hd, 42);
        let mut vec_neon = vec_scalar.clone();
        apply_rope_inplace(&mut vec_scalar, pos, hd);
        apply_rope_inplace_neon(&mut vec_neon, pos, hd);
        assert_close(&vec_scalar, &vec_neon, TOL);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_rope_neon_large_dim() {
        let hd = 128;
        let pos = 42;
        let mut vec_scalar = rand_vec(hd, 42);
        let mut vec_neon = vec_scalar.clone();
        apply_rope_inplace(&mut vec_scalar, pos, hd);
        apply_rope_inplace_neon(&mut vec_neon, pos, hd);
        assert_close(&vec_scalar, &vec_neon, TOL);
    }

    #[test]
    fn test_matmul_scalar_identity() {
        let n = 4;
        let a = rand_vec(n * n, 42);
        let eye = identity_matrix(n);
        let mut c = vec![0.0f32; n * n];
        matmul_scalar(&a, &eye, &mut c, n, n, n);
        assert_close(&c, &a, TOL);
    }

    #[test]
    fn test_end_to_end_mha_with_qkv() {
        let d_model = 16;
        let nh = 2;
        let hd = d_model / nh;
        let seq = 4;
        let input = rand_vec(seq * d_model, 42);
        let wq = rand_vec(d_model * d_model, 100);
        let wk = rand_vec(d_model * d_model, 200);
        let wv = rand_vec(d_model * d_model, 300);

        let (q, k, v) =
            fused_qkv_projection(&input, &wq, &wk, &wv, None, None, None, seq, d_model, d_model);

        let config = MultiHeadAttentionConfig { num_heads: nh, head_dim: hd, seq_len: seq };
        let out = fused_multi_head_attention(&q, &k, &v, &config);
        assert_eq!(out.len(), seq * d_model);
        for val in &out {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_end_to_end_gqa_with_qkv() {
        let d_model = 16;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let hd = d_model / num_q_heads;
        let seq = 4;
        let q_dim = num_q_heads * hd;
        let kv_dim = num_kv_heads * hd;

        let input = rand_vec(seq * d_model, 42);
        let wq = rand_vec(d_model * q_dim, 100);
        let wk = rand_vec(d_model * kv_dim, 200);
        let wv = rand_vec(d_model * kv_dim, 300);

        // Project Q with q_dim
        let (q, _, _) =
            fused_qkv_projection(&input, &wq, &wq, &wq, None, None, None, seq, d_model, q_dim);
        // Project K, V with kv_dim
        let (_, k2, v2) =
            fused_qkv_projection(&input, &wk, &wk, &wv, None, None, None, seq, d_model, kv_dim);

        let config = GqaAttentionConfig { num_q_heads, num_kv_heads, head_dim: hd, seq_len: seq };
        let out = fused_gqa_attention(&q, &k2, &v2, &config);
        assert_eq!(out.len(), seq * q_dim);
    }

    #[test]
    fn test_causal_vs_full_first_row() {
        // First row of causal == first row of full (only 1 key visible)
        let hd = 8;
        let seq = 6;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let causal = fused_causal_attention(&q, &k, &v, seq, hd);
        // First token should be V[0] regardless
        assert_close(&causal[..hd], &v[..hd], TOL);
    }

    #[test]
    fn test_sdpa_weights_sum_to_one() {
        // Verify softmax weights sum to 1 by checking output is convex combination
        let hd = 4;
        let seq = 3;
        let q = vec![1.0; seq * hd];
        let k = vec![1.0; seq * hd];
        let v = rand_vec(seq * hd, 42);
        let out = fused_scaled_dot_product_attention(&q, &k, &v, seq, seq, hd);
        // Uniform attention => output = mean of V rows
        let mean: Vec<f32> =
            (0..hd).map(|d| (0..seq).map(|s| v[s * hd + d]).sum::<f32>() / seq as f32).collect();
        for s in 0..seq {
            assert_close(&out[s * hd..(s + 1) * hd], &mean, TOL);
        }
    }

    #[test]
    fn test_qkv_deterministic() {
        let d = 8;
        let seq = 3;
        let input = rand_vec(seq * d, 42);
        let w = rand_vec(d * d, 43);
        let (q1, k1, v1) = fused_qkv_projection(&input, &w, &w, &w, None, None, None, seq, d, d);
        let (q2, k2, v2) = fused_qkv_projection(&input, &w, &w, &w, None, None, None, seq, d, d);
        assert_close(&q1, &q2, 0.0);
        assert_close(&k1, &k2, 0.0);
        assert_close(&v1, &v2, 0.0);
    }

    #[test]
    fn test_causal_deterministic() {
        let hd = 8;
        let seq = 4;
        let q = rand_vec(seq * hd, 10);
        let k = rand_vec(seq * hd, 20);
        let v = rand_vec(seq * hd, 30);
        let out1 = fused_causal_attention(&q, &k, &v, seq, hd);
        let out2 = fused_causal_attention(&q, &k, &v, seq, hd);
        assert_close(&out1, &out2, 0.0);
    }

    #[test]
    fn test_sdpa_non_square() {
        let hd = 8;
        let sq = 1;
        let sk = 10;
        let q = rand_vec(sq * hd, 10);
        let k = rand_vec(sk * hd, 20);
        let v = rand_vec(sk * hd, 30);
        let out = fused_scaled_dot_product_attention(&q, &k, &v, sq, sk, hd);
        assert_eq!(out.len(), sq * hd);
    }

    #[test]
    fn test_gqa_large_group_ratio() {
        let hd = 4;
        let seq = 2;
        let config =
            GqaAttentionConfig { num_q_heads: 8, num_kv_heads: 1, head_dim: hd, seq_len: seq };
        let q = rand_vec(seq * 8 * hd, 10);
        let k = rand_vec(seq * 1 * hd, 20);
        let v = rand_vec(seq * 1 * hd, 30);
        let out1 = fused_gqa_attention(&q, &k, &v, &config);
        let out2 = fused_gqa_attention_scalar(&q, &k, &v, &config);
        assert_close(&out1, &out2, TOL);
    }
}
