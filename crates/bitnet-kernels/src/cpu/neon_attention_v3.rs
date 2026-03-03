//! ARM NEON advanced attention v3 kernels for Apple Silicon (aarch64).
//!
//! Provides six attention operations with NEON-accelerated inner loops
//! and scalar fallbacks:
//!
//! 1. Scaled dot-product attention (single-head)
//! 2. Multi-head attention
//! 3. Causal (auto-regressive) attention with triangular mask
//! 4. Grouped-query attention (GQA)
//! 5. Flash (tiled/blocked) attention for cache efficiency
//! 6. Attention with ALiBi positional bias

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::derivable_impls,
    clippy::excessive_precision,
    clippy::manual_is_multiple_of,
    clippy::manual_memcpy,
    dead_code,
    unused_unsafe
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── Softmax helpers ────────────────────────────────────────────────────

/// Scalar softmax in-place using max-subtract-exp-normalize.
fn scalar_softmax_inplace(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in data.iter_mut() {
            *v /= sum;
        }
    }
}

/// NEON-accelerated softmax in-place.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_inplace(data: &mut [f32]) {
    let len = data.len();
    if len == 0 {
        return;
    }

    // Phase 1: find max
    let ptr = data.as_ptr();
    let chunks = len / LANES;
    let _remainder = len % LANES;

    let mut vmax = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        vmax = unsafe { vmaxq_f32(vmax, v) };
    }
    let mut max_val = unsafe { vmaxvq_f32(vmax) };
    for i in (chunks * LANES)..len {
        max_val = max_val.max(data[i]);
    }

    // Phase 2: exp(x - max)
    let vmax_scalar = unsafe { vdupq_n_f32(max_val) };
    let mut vsum = unsafe { vdupq_n_f32(0.0) };
    let out_ptr = data.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = unsafe { vsubq_f32(v, vmax_scalar) };
        // Scalar exp per lane (no NEON exp intrinsic).
        let mut arr = [0.0f32; LANES];
        unsafe { vst1q_f32(arr.as_mut_ptr(), shifted) };
        for a in &mut arr {
            *a = a.exp();
        }
        let exp_v = unsafe { vld1q_f32(arr.as_ptr()) };
        vsum = unsafe { vaddq_f32(vsum, exp_v) };
        unsafe { vst1q_f32(out_ptr.add(i * LANES), exp_v) };
    }
    let mut sum = unsafe { vaddvq_f32(vsum) };
    for i in (chunks * LANES)..len {
        let e = (data[i] - max_val).exp();
        data[i] = e;
        sum += e;
    }

    // Phase 3: normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        let vinv = unsafe { vdupq_n_f32(inv_sum) };
        for i in 0..chunks {
            let v = unsafe { vld1q_f32(out_ptr.add(i * LANES) as *const f32) };
            let normed = unsafe { vmulq_f32(v, vinv) };
            unsafe { vst1q_f32(out_ptr.add(i * LANES), normed) };
        }
        for i in (chunks * LANES)..len {
            data[i] *= inv_sum;
        }
    }
}

// ── NEON dot product helper ────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let chunks = len / LANES;
    let mut vacc = unsafe { vdupq_n_f32(0.0) };
    for i in 0..chunks {
        let va = unsafe { vld1q_f32(a.as_ptr().add(i * LANES)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i * LANES)) };
        vacc = unsafe { vfmaq_f32(vacc, va, vb) };
    }
    let mut acc = unsafe { vaddvq_f32(vacc) };
    for i in (chunks * LANES)..len {
        acc += a[i] * b[i];
    }
    acc
}

fn scalar_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..len {
        acc += a[i] * b[i];
    }
    acc
}

// ── NEON weighted accumulate helper ────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_weighted_acc(output: &mut [f32], src: &[f32], weight: f32, len: usize) {
    let chunks = len / LANES;
    let vw = unsafe { vdupq_n_f32(weight) };
    let out_ptr = output.as_mut_ptr();
    for i in 0..chunks {
        let vo = unsafe { vld1q_f32(out_ptr.add(i * LANES) as *const f32) };
        let vs = unsafe { vld1q_f32(src.as_ptr().add(i * LANES)) };
        let result = unsafe { vfmaq_f32(vo, vs, vw) };
        unsafe { vst1q_f32(out_ptr.add(i * LANES), result) };
    }
    for i in (chunks * LANES)..len {
        output[i] += src[i] * weight;
    }
}

fn scalar_weighted_acc(output: &mut [f32], src: &[f32], weight: f32, len: usize) {
    for i in 0..len {
        output[i] += src[i] * weight;
    }
}

// ════════════════════════════════════════════════════════════════════════
// 1. Scaled dot-product attention (single-head)
// ════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scaled_dot_product_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        // Compute scores for row i
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = unsafe { neon_dot_f32(q_row, k_row, head_dim) } * scale;
        }
        // Softmax
        unsafe { neon_softmax_inplace(&mut scores) };
        // Weighted sum of V
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        for j in 0..seq_len {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            unsafe { neon_weighted_acc(out_row, v_row, scores[j], head_dim) };
        }
    }
}

fn scalar_scaled_dot_product_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = scalar_dot_f32(q_row, k_row, head_dim) * scale;
        }
        scalar_softmax_inplace(&mut scores);
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        for j in 0..seq_len {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            scalar_weighted_acc(out_row, v_row, scores[j], head_dim);
        }
    }
}

/// Single-head scaled dot-product attention: softmax(QK^T / scale) * V.
pub fn scaled_dot_product_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection above guarantees NEON is available.
            unsafe {
                neon_scaled_dot_product_attention_f32(q, k, v, output, seq_len, head_dim, scale);
            }
            return;
        }
    }
    scalar_scaled_dot_product_attention_f32(q, k, v, output, seq_len, head_dim, scale);
}

// ════════════════════════════════════════════════════════════════════════
// 2. Multi-head attention
// ════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_multi_head_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_size = seq_len * head_dim;
    for h in 0..num_heads {
        let offset = h * head_size;
        let q_head = &q[offset..offset + head_size];
        let k_head = &k[offset..offset + head_size];
        let v_head = &v[offset..offset + head_size];
        let out_head = &mut output[offset..offset + head_size];
        unsafe {
            neon_scaled_dot_product_attention_f32(
                q_head, k_head, v_head, out_head, seq_len, head_dim, scale,
            );
        }
    }
}

fn scalar_multi_head_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_size = seq_len * head_dim;
    for h in 0..num_heads {
        let offset = h * head_size;
        let q_head = &q[offset..offset + head_size];
        let k_head = &k[offset..offset + head_size];
        let v_head = &v[offset..offset + head_size];
        let out_head = &mut output[offset..offset + head_size];
        scalar_scaled_dot_product_attention_f32(
            q_head, k_head, v_head, out_head, seq_len, head_dim, scale,
        );
    }
}

/// Multi-head attention: each head gets its own QKV slice.
pub fn multi_head_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_multi_head_attention_f32(q, k, v, output, num_heads, seq_len, head_dim);
            }
            return;
        }
    }
    scalar_multi_head_attention_f32(q, k, v, output, num_heads, seq_len, head_dim);
}

// ════════════════════════════════════════════════════════════════════════
// 3. Causal (auto-regressive) attention
// ════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_causal_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        // Only attend to positions 0..=i (causal mask)
        let valid = i + 1;
        for j in 0..valid {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = unsafe { neon_dot_f32(q_row, k_row, head_dim) } * scale;
        }
        // Softmax only over valid positions
        unsafe { neon_softmax_inplace(&mut scores[..valid]) };
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        for j in 0..valid {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            unsafe { neon_weighted_acc(out_row, v_row, scores[j], head_dim) };
        }
    }
}

fn scalar_causal_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let valid = i + 1;
        for j in 0..valid {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = scalar_dot_f32(q_row, k_row, head_dim) * scale;
        }
        scalar_softmax_inplace(&mut scores[..valid]);
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        for j in 0..valid {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            scalar_weighted_acc(out_row, v_row, scores[j], head_dim);
        }
    }
}

/// Causal attention with lower-triangular mask.
pub fn causal_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_causal_attention_f32(q, k, v, output, seq_len, head_dim, scale);
            }
            return;
        }
    }
    scalar_causal_attention_f32(q, k, v, output, seq_len, head_dim, scale);
}

// ════════════════════════════════════════════════════════════════════════
// 4. Grouped-query attention (GQA)
// ════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_grouped_query_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_head_size = seq_len * head_dim;
    let kv_head_size = seq_len * head_dim;
    let heads_per_group = num_q_heads / num_kv_heads;

    for qh in 0..num_q_heads {
        let kv_idx = qh / heads_per_group;
        let q_off = qh * q_head_size;
        let kv_off = kv_idx * kv_head_size;
        let q_head = &q[q_off..q_off + q_head_size];
        let k_head = &k[kv_off..kv_off + kv_head_size];
        let v_head = &v[kv_off..kv_off + kv_head_size];
        let out_head = &mut output[q_off..q_off + q_head_size];
        unsafe {
            neon_scaled_dot_product_attention_f32(
                q_head, k_head, v_head, out_head, seq_len, head_dim, scale,
            );
        }
    }
}

fn scalar_grouped_query_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_head_size = seq_len * head_dim;
    let kv_head_size = seq_len * head_dim;
    let heads_per_group = num_q_heads / num_kv_heads;

    for qh in 0..num_q_heads {
        let kv_idx = qh / heads_per_group;
        let q_off = qh * q_head_size;
        let kv_off = kv_idx * kv_head_size;
        let q_head = &q[q_off..q_off + q_head_size];
        let k_head = &k[kv_off..kv_off + kv_head_size];
        let v_head = &v[kv_off..kv_off + kv_head_size];
        let out_head = &mut output[q_off..q_off + q_head_size];
        scalar_scaled_dot_product_attention_f32(
            q_head, k_head, v_head, out_head, seq_len, head_dim, scale,
        );
    }
}

/// Grouped-query attention: Q heads share KV heads in groups.
pub fn grouped_query_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_grouped_query_attention_f32(
                    q,
                    k,
                    v,
                    output,
                    num_q_heads,
                    num_kv_heads,
                    seq_len,
                    head_dim,
                );
            }
            return;
        }
    }
    scalar_grouped_query_attention_f32(
        q,
        k,
        v,
        output,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
    );
}

// ════════════════════════════════════════════════════════════════════════
// 5. Flash (tiled/blocked) attention
// ════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_flash_attention_tiled_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let bs = block_size.max(1);

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut rescale_out = false;

        // Process key/value in blocks
        let num_blocks = (seq_len + bs - 1) / bs;
        for blk in 0..num_blocks {
            let j_start = blk * bs;
            let j_end = (j_start + bs).min(seq_len);
            let blk_len = j_end - j_start;

            // Compute scores for this block
            let mut block_scores = vec![0.0f32; blk_len];
            for (idx, j) in (j_start..j_end).enumerate() {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                block_scores[idx] = unsafe { neon_dot_f32(q_row, k_row, head_dim) } * scale;
            }

            // Block max
            let block_max = block_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let new_max = running_max.max(block_max);

            // Rescale existing output
            if rescale_out {
                let correction = (running_max - new_max).exp();
                let vcorr = unsafe { vdupq_n_f32(correction) };
                let chunks = head_dim / LANES;
                let optr = out_row.as_mut_ptr();
                for c in 0..chunks {
                    let vo = unsafe { vld1q_f32(optr.add(c * LANES) as *const f32) };
                    let scaled = unsafe { vmulq_f32(vo, vcorr) };
                    unsafe { vst1q_f32(optr.add(c * LANES), scaled) };
                }
                for d in (chunks * LANES)..head_dim {
                    out_row[d] *= correction;
                }
                running_sum *= correction;
            }

            // Accumulate exp(score - new_max) * V
            for (idx, j) in (j_start..j_end).enumerate() {
                let w = (block_scores[idx] - new_max).exp();
                running_sum += w;
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                unsafe { neon_weighted_acc(out_row, v_row, w, head_dim) };
            }
            running_max = new_max;
            rescale_out = true;
        }

        // Final normalization
        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            let vinv = unsafe { vdupq_n_f32(inv) };
            let chunks = head_dim / LANES;
            let optr = out_row.as_mut_ptr();
            for c in 0..chunks {
                let vo = unsafe { vld1q_f32(optr.add(c * LANES) as *const f32) };
                let normed = unsafe { vmulq_f32(vo, vinv) };
                unsafe { vst1q_f32(optr.add(c * LANES), normed) };
            }
            for d in (chunks * LANES)..head_dim {
                out_row[d] *= inv;
            }
        }
    }
}

fn scalar_flash_attention_tiled_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let bs = block_size.max(1);

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut rescale_out = false;

        let num_blocks = (seq_len + bs - 1) / bs;
        for blk in 0..num_blocks {
            let j_start = blk * bs;
            let j_end = (j_start + bs).min(seq_len);
            let blk_len = j_end - j_start;

            let mut block_scores = vec![0.0f32; blk_len];
            for (idx, j) in (j_start..j_end).enumerate() {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                block_scores[idx] = scalar_dot_f32(q_row, k_row, head_dim) * scale;
            }

            let block_max = block_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let new_max = running_max.max(block_max);

            if rescale_out {
                let correction = (running_max - new_max).exp();
                for d in 0..head_dim {
                    out_row[d] *= correction;
                }
                running_sum *= correction;
            }

            for (idx, j) in (j_start..j_end).enumerate() {
                let w = (block_scores[idx] - new_max).exp();
                running_sum += w;
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                scalar_weighted_acc(out_row, v_row, w, head_dim);
            }
            running_max = new_max;
            rescale_out = true;
        }

        if running_sum > 0.0 {
            let inv = 1.0 / running_sum;
            for d in 0..head_dim {
                out_row[d] *= inv;
            }
        }
    }
}

/// Tiled/blocked attention for improved cache utilization.
pub fn flash_attention_tiled_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_flash_attention_tiled_f32(q, k, v, output, seq_len, head_dim, block_size);
            }
            return;
        }
    }
    scalar_flash_attention_tiled_f32(q, k, v, output, seq_len, head_dim, block_size);
}

// ════════════════════════════════════════════════════════════════════════
// 6. Attention with ALiBi positional bias
// ════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_with_alibi_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    alibi_slope: f32,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot = unsafe { neon_dot_f32(q_row, k_row, head_dim) } * scale;
            // ALiBi bias: slope * -|i - j|
            let distance = (i as f32 - j as f32).abs();
            scores[j] = dot - alibi_slope * distance;
        }
        unsafe { neon_softmax_inplace(&mut scores) };
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        for j in 0..seq_len {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            unsafe { neon_weighted_acc(out_row, v_row, scores[j], head_dim) };
        }
    }
}

fn scalar_attention_with_alibi_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    alibi_slope: f32,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..seq_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot = scalar_dot_f32(q_row, k_row, head_dim) * scale;
            let distance = (i as f32 - j as f32).abs();
            scores[j] = dot - alibi_slope * distance;
        }
        scalar_softmax_inplace(&mut scores);
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        out_row.fill(0.0);
        for j in 0..seq_len {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            scalar_weighted_acc(out_row, v_row, scores[j], head_dim);
        }
    }
}

/// Attention with ALiBi (Attention with Linear Biases) positional encoding.
pub fn attention_with_alibi_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    alibi_slope: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_with_alibi_f32(q, k, v, output, seq_len, head_dim, alibi_slope);
            }
            return;
        }
    }
    scalar_attention_with_alibi_f32(q, k, v, output, seq_len, head_dim, alibi_slope);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── f64 naive reference implementations ────────────────────────────

    fn ref_softmax_f64(data: &mut [f64]) {
        if data.is_empty() {
            return;
        }
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for v in data.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in data.iter_mut() {
                *v /= sum;
            }
        }
    }

    fn ref_sdpa_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * head_dim];
        let mut scores = vec![0.0f64; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
                }
                scores[j] = dot * scale as f64;
            }
            ref_softmax_f64(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for j in 0..seq_len {
                    acc += scores[j] * v[j * head_dim + d] as f64;
                }
                output[i * head_dim + d] = acc as f32;
            }
        }
        output
    }

    fn ref_causal_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            let valid = i + 1;
            let mut scores = vec![0.0f64; valid];
            for j in 0..valid {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
                }
                scores[j] = dot * scale as f64;
            }
            ref_softmax_f64(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for j in 0..valid {
                    acc += scores[j] * v[j * head_dim + d] as f64;
                }
                output[i * head_dim + d] = acc as f32;
            }
        }
        output
    }

    fn ref_alibi_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        alibi_slope: f32,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut output = vec![0.0f32; seq_len * head_dim];
        let mut scores = vec![0.0f64; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
                }
                let dist = ((i as i64) - (j as i64)).unsigned_abs() as f64;
                scores[j] = dot * scale - alibi_slope as f64 * dist;
            }
            ref_softmax_f64(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0f64;
                for j in 0..seq_len {
                    acc += scores[j] * v[j * head_dim + d] as f64;
                }
                output[i * head_dim + d] = acc as f32;
            }
        }
        output
    }

    // ── Deterministic data generation ──────────────────────────────────

    fn make_data(len: usize, seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(((s >> 16) as f32 / 32768.0) - 1.0);
        }
        v
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(diff < tol, "{label}[{i}]: {x} vs {y}, diff={diff} > tol={tol}");
        }
    }

    // ── 1. Scaled dot-product attention tests ──────────────────────────

    #[test]
    fn test_sdpa_basic_correctness() {
        let (seq_len, head_dim) = (4, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 1);
        let k = make_data(seq_len * head_dim, 2);
        let v = make_data(seq_len * head_dim, 3);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "sdpa_basic");
    }

    #[test]
    fn test_sdpa_seq_len_1() {
        let (seq_len, head_dim) = (1, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 10);
        let k = make_data(seq_len * head_dim, 11);
        let v = make_data(seq_len * head_dim, 12);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        // With seq_len=1, output == v (softmax of single element is 1.0)
        assert_close(&out, &v, 1e-6, "sdpa_seq1");
    }

    #[test]
    fn test_sdpa_head_dim_1() {
        let (seq_len, head_dim) = (4, 1);
        let scale = 1.0;
        let q = make_data(seq_len * head_dim, 20);
        let k = make_data(seq_len * head_dim, 21);
        let v = make_data(seq_len * head_dim, 22);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "sdpa_hd1");
    }

    #[test]
    fn test_sdpa_large_dim() {
        let (seq_len, head_dim) = (8, 64);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 30);
        let k = make_data(seq_len * head_dim, 31);
        let v = make_data(seq_len * head_dim, 32);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-4, "sdpa_large");
    }

    #[test]
    fn test_sdpa_identity_qk() {
        // When Q == K, attention weights should be roughly uniform (same self-similarity)
        let (seq_len, head_dim) = (3, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = vec![1.0; seq_len * head_dim];
        let k = q.clone();
        let v = make_data(seq_len * head_dim, 40);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        // All Q rows identical → all attend uniformly → output should be mean of V rows
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "sdpa_identity");
    }

    #[test]
    fn test_sdpa_numerical_stability_large_values() {
        let (seq_len, head_dim) = (4, 4);
        let scale = 1.0;
        let q = vec![100.0f32; seq_len * head_dim];
        let k = vec![100.0f32; seq_len * head_dim];
        let v = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        // Should not produce NaN/Inf
        for val in &out {
            assert!(val.is_finite(), "large values produced non-finite: {val}");
        }
    }

    #[test]
    fn test_sdpa_numerical_stability_small_values() {
        let (seq_len, head_dim) = (4, 4);
        let scale = 1.0;
        let q = vec![1e-10f32; seq_len * head_dim];
        let k = vec![1e-10f32; seq_len * head_dim];
        let v = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        for val in &out {
            assert!(val.is_finite(), "small values produced non-finite: {val}");
        }
    }

    #[test]
    fn test_sdpa_weights_sum_to_one() {
        let (seq_len, head_dim) = (5, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 50);
        let k = make_data(seq_len * head_dim, 51);
        // Use identity V to recover attention weights per row
        // Compute with V = ones and check output sums
        let v_ones = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v_ones, &mut out, seq_len, head_dim, scale);
        // Since V is all 1.0, output[i][d] = sum_j(attn[i][j]) * 1.0 = 1.0
        for val in &out {
            assert!((*val - 1.0).abs() < 1e-5, "weight sum check: got {val}, expected ~1.0");
        }
    }

    #[test]
    fn test_sdpa_different_scales() {
        let (seq_len, head_dim) = (3, 4);
        let q = make_data(seq_len * head_dim, 60);
        let k = make_data(seq_len * head_dim, 61);
        let v = make_data(seq_len * head_dim, 62);
        for &scale in &[0.1, 0.5, 1.0, 2.0, 10.0] {
            let mut out = vec![0.0f32; seq_len * head_dim];
            scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
            let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
            assert_close(&out, &expected, 1e-4, &format!("sdpa_scale_{scale}"));
        }
    }

    #[test]
    fn test_sdpa_non_power_of_2_dim() {
        let (seq_len, head_dim) = (3, 7);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 70);
        let k = make_data(seq_len * head_dim, 71);
        let v = make_data(seq_len * head_dim, 72);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "sdpa_dim7");
    }

    // ── 2. Multi-head attention tests ──────────────────────────────────

    #[test]
    fn test_mha_basic() {
        let (num_heads, seq_len, head_dim) = (2, 3, 4);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 100);
        let k = make_data(total, 101);
        let v = make_data(total, 102);
        let mut out = vec![0.0f32; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, num_heads, seq_len, head_dim);
        // Verify each head independently
        let scale = 1.0 / (head_dim as f32).sqrt();
        let hs = seq_len * head_dim;
        for h in 0..num_heads {
            let expected = ref_sdpa_f64(
                &q[h * hs..(h + 1) * hs],
                &k[h * hs..(h + 1) * hs],
                &v[h * hs..(h + 1) * hs],
                seq_len,
                head_dim,
                scale,
            );
            assert_close(&out[h * hs..(h + 1) * hs], &expected, 1e-5, &format!("mha_head{h}"));
        }
    }

    #[test]
    fn test_mha_single_head() {
        let (num_heads, seq_len, head_dim) = (1, 4, 8);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 110);
        let k = make_data(total, 111);
        let v = make_data(total, 112);
        let mut out_mha = vec![0.0f32; total];
        let mut out_sdpa = vec![0.0f32; total];
        let scale = 1.0 / (head_dim as f32).sqrt();
        multi_head_attention_f32(&q, &k, &v, &mut out_mha, num_heads, seq_len, head_dim);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_sdpa, seq_len, head_dim, scale);
        assert_close(&out_mha, &out_sdpa, 1e-6, "mha_single_vs_sdpa");
    }

    #[test]
    fn test_mha_many_heads() {
        let (num_heads, seq_len, head_dim) = (8, 4, 16);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 120);
        let k = make_data(total, 121);
        let v = make_data(total, 122);
        let mut out = vec![0.0f32; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, num_heads, seq_len, head_dim);
        for val in &out {
            assert!(val.is_finite(), "mha_many: non-finite output");
        }
    }

    #[test]
    fn test_mha_seq1() {
        let (num_heads, seq_len, head_dim) = (4, 1, 8);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 130);
        let k = make_data(total, 131);
        let v = make_data(total, 132);
        let mut out = vec![0.0f32; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, num_heads, seq_len, head_dim);
        // seq_len=1 → output == v for each head
        assert_close(&out, &v, 1e-6, "mha_seq1");
    }

    #[test]
    fn test_mha_head_independence() {
        let (num_heads, seq_len, head_dim) = (2, 3, 4);
        let hs = seq_len * head_dim;
        let total = num_heads * hs;
        let q = make_data(total, 140);
        let k = make_data(total, 141);
        let v = make_data(total, 142);
        let mut out = vec![0.0f32; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, num_heads, seq_len, head_dim);
        // Modify head 0's input and verify head 1 is unchanged
        let mut q2 = q.clone();
        q2[0] += 999.0;
        let mut out2 = vec![0.0f32; total];
        multi_head_attention_f32(&q2, &k, &v, &mut out2, num_heads, seq_len, head_dim);
        // Head 1 should be identical
        assert_close(&out[hs..], &out2[hs..], 1e-7, "mha_head_independence");
    }

    // ── 3. Causal attention tests ──────────────────────────────────────

    #[test]
    fn test_causal_basic() {
        let (seq_len, head_dim) = (4, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 200);
        let k = make_data(seq_len * head_dim, 201);
        let v = make_data(seq_len * head_dim, 202);
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_causal_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "causal_basic");
    }

    #[test]
    fn test_causal_seq1() {
        let (seq_len, head_dim) = (1, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 210);
        let k = make_data(seq_len * head_dim, 211);
        let v = make_data(seq_len * head_dim, 212);
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        // seq_len=1 → identical to full attention
        assert_close(&out, &v, 1e-6, "causal_seq1");
    }

    #[test]
    fn test_causal_mask_lower_triangle() {
        // Position 0 can only attend to position 0
        // Position 1 can attend to 0,1
        // etc.
        // Verify by using V with distinct rows and checking first row output == V[0]
        let (seq_len, head_dim) = (4, 2);
        let scale = 1.0;
        let q = vec![0.0f32; seq_len * head_dim]; // zero queries
        let k = vec![0.0f32; seq_len * head_dim]; // zero keys → equal scores
        let mut v = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                v[i * head_dim + d] = (i * head_dim + d) as f32;
            }
        }
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        // Row 0: attends only to V[0], so out[0] == V[0]
        assert_close(&out[0..head_dim], &v[0..head_dim], 1e-6, "causal_row0");
        // Row 1: uniform over V[0],V[1] → mean
        for d in 0..head_dim {
            let expected = (v[d] + v[head_dim + d]) / 2.0;
            assert!(
                (out[head_dim + d] - expected).abs() < 1e-5,
                "causal_row1[{d}]: {} vs {}",
                out[head_dim + d],
                expected
            );
        }
    }

    #[test]
    fn test_causal_differs_from_full() {
        let (seq_len, head_dim) = (4, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 220);
        let k = make_data(seq_len * head_dim, 221);
        let v = make_data(seq_len * head_dim, 222);
        let mut out_causal = vec![0.0f32; seq_len * head_dim];
        let mut out_full = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out_causal, seq_len, head_dim, scale);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_full, seq_len, head_dim, scale);
        // Row 0: causal attends only to pos 0 while full attends to all → differ
        let row0_diff: f32 = (0..head_dim).map(|d| (out_causal[d] - out_full[d]).abs()).sum();
        assert!(row0_diff > 1e-3, "row0 should differ: diff={row0_diff}");
        // Last row: causal attends to all → should equal full
        let last_start = (seq_len - 1) * head_dim;
        assert_close(
            &out_causal[last_start..last_start + head_dim],
            &out_full[last_start..last_start + head_dim],
            1e-6,
            "causal_vs_full_last_row",
        );
    }

    #[test]
    fn test_causal_numerical_stability() {
        let (seq_len, head_dim) = (4, 4);
        let scale = 1.0;
        let q = vec![50.0f32; seq_len * head_dim];
        let k = vec![50.0f32; seq_len * head_dim];
        let v = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        for val in &out {
            assert!(val.is_finite(), "causal stability: non-finite {val}");
        }
    }

    #[test]
    fn test_causal_weights_sum_to_one() {
        let (seq_len, head_dim) = (5, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 230);
        let k = make_data(seq_len * head_dim, 231);
        let v_ones = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v_ones, &mut out, seq_len, head_dim, scale);
        for val in &out {
            assert!((*val - 1.0).abs() < 1e-5, "causal weight sum: {val} != 1.0");
        }
    }

    #[test]
    fn test_causal_head_dim_1() {
        let (seq_len, head_dim) = (3, 1);
        let scale = 1.0;
        let q = make_data(seq_len * head_dim, 235);
        let k = make_data(seq_len * head_dim, 236);
        let v = make_data(seq_len * head_dim, 237);
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_causal_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "causal_hd1");
    }

    // ── 4. Grouped-query attention tests ───────────────────────────────

    #[test]
    fn test_gqa_basic() {
        let (num_q, num_kv, seq_len, head_dim) = (4, 2, 3, 8);
        let q = make_data(num_q * seq_len * head_dim, 300);
        let k = make_data(num_kv * seq_len * head_dim, 301);
        let v = make_data(num_kv * seq_len * head_dim, 302);
        let mut out = vec![0.0f32; num_q * seq_len * head_dim];
        grouped_query_attention_f32(&q, &k, &v, &mut out, num_q, num_kv, seq_len, head_dim);
        for val in &out {
            assert!(val.is_finite(), "gqa_basic: non-finite");
        }
    }

    #[test]
    fn test_gqa_head_sharing() {
        // 4 Q heads, 2 KV heads → heads 0,1 share KV[0], heads 2,3 share KV[1]
        let (num_q, num_kv, seq_len, head_dim) = (4, 2, 3, 4);
        let hs = seq_len * head_dim;
        // Make Q heads 0 and 1 identical
        let mut q = make_data(num_q * hs, 310);
        let head0: Vec<f32> = q[0..hs].to_vec();
        q[hs..2 * hs].copy_from_slice(&head0);
        let k = make_data(num_kv * hs, 311);
        let v = make_data(num_kv * hs, 312);
        let mut out = vec![0.0f32; num_q * hs];
        grouped_query_attention_f32(&q, &k, &v, &mut out, num_q, num_kv, seq_len, head_dim);
        // Heads 0 and 1 share KV[0] and have same Q → identical output
        assert_close(&out[0..hs], &out[hs..2 * hs], 1e-6, "gqa_sharing_01");
    }

    #[test]
    fn test_gqa_equal_heads_is_mha() {
        // When num_q == num_kv, GQA == MHA
        let (num_heads, seq_len, head_dim) = (2, 3, 4);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 320);
        let k = make_data(total, 321);
        let v = make_data(total, 322);
        let mut out_gqa = vec![0.0f32; total];
        let mut out_mha = vec![0.0f32; total];
        grouped_query_attention_f32(
            &q,
            &k,
            &v,
            &mut out_gqa,
            num_heads,
            num_heads,
            seq_len,
            head_dim,
        );
        multi_head_attention_f32(&q, &k, &v, &mut out_mha, num_heads, seq_len, head_dim);
        assert_close(&out_gqa, &out_mha, 1e-6, "gqa_eq_mha");
    }

    #[test]
    fn test_gqa_single_kv_head() {
        // All Q heads share one KV head (multi-query attention)
        let (num_q, num_kv, seq_len, head_dim) = (4, 1, 3, 4);
        let q_total = num_q * seq_len * head_dim;
        let kv_total = num_kv * seq_len * head_dim;
        let q = make_data(q_total, 330);
        let k = make_data(kv_total, 331);
        let v = make_data(kv_total, 332);
        let mut out = vec![0.0f32; q_total];
        grouped_query_attention_f32(&q, &k, &v, &mut out, num_q, num_kv, seq_len, head_dim);
        // Verify all heads use the same KV → independently correct
        let scale = 1.0 / (head_dim as f32).sqrt();
        let hs = seq_len * head_dim;
        for h in 0..num_q {
            let expected = ref_sdpa_f64(
                &q[h * hs..(h + 1) * hs],
                &k[0..hs],
                &v[0..hs],
                seq_len,
                head_dim,
                scale,
            );
            assert_close(
                &out[h * hs..(h + 1) * hs],
                &expected,
                1e-5,
                &format!("gqa_single_kv_head{h}"),
            );
        }
    }

    #[test]
    fn test_gqa_seq1() {
        let (num_q, num_kv, seq_len, head_dim) = (4, 2, 1, 8);
        let q = make_data(num_q * seq_len * head_dim, 340);
        let k = make_data(num_kv * seq_len * head_dim, 341);
        let v = make_data(num_kv * seq_len * head_dim, 342);
        let mut out = vec![0.0f32; num_q * seq_len * head_dim];
        grouped_query_attention_f32(&q, &k, &v, &mut out, num_q, num_kv, seq_len, head_dim);
        // seq_len=1: each head's output == its corresponding V
        let hs = seq_len * head_dim;
        let hpg = num_q / num_kv;
        for qh in 0..num_q {
            let kv_idx = qh / hpg;
            assert_close(
                &out[qh * hs..(qh + 1) * hs],
                &v[kv_idx * hs..(kv_idx + 1) * hs],
                1e-6,
                &format!("gqa_seq1_head{qh}"),
            );
        }
    }

    // ── 5. Flash/tiled attention tests ─────────────────────────────────

    #[test]
    fn test_flash_matches_sdpa() {
        let (seq_len, head_dim) = (6, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 400);
        let k = make_data(seq_len * head_dim, 401);
        let v = make_data(seq_len * head_dim, 402);
        let mut out_flash = vec![0.0f32; seq_len * head_dim];
        let mut out_sdpa = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, 2);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_sdpa, seq_len, head_dim, scale);
        assert_close(&out_flash, &out_sdpa, 1e-4, "flash_vs_sdpa");
    }

    #[test]
    fn test_flash_block_size_1() {
        let (seq_len, head_dim) = (4, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 410);
        let k = make_data(seq_len * head_dim, 411);
        let v = make_data(seq_len * head_dim, 412);
        let mut out_flash = vec![0.0f32; seq_len * head_dim];
        let mut out_ref = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, 1);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_ref, seq_len, head_dim, scale);
        assert_close(&out_flash, &out_ref, 1e-4, "flash_bs1");
    }

    #[test]
    fn test_flash_block_size_larger_than_seq() {
        let (seq_len, head_dim) = (3, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 420);
        let k = make_data(seq_len * head_dim, 421);
        let v = make_data(seq_len * head_dim, 422);
        let mut out_flash = vec![0.0f32; seq_len * head_dim];
        let mut out_ref = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, 64);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_ref, seq_len, head_dim, scale);
        assert_close(&out_flash, &out_ref, 1e-5, "flash_big_bs");
    }

    #[test]
    fn test_flash_various_block_sizes() {
        let (seq_len, head_dim) = (8, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 430);
        let k = make_data(seq_len * head_dim, 431);
        let v = make_data(seq_len * head_dim, 432);
        let mut out_ref = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_ref, seq_len, head_dim, scale);
        for bs in [1, 2, 3, 4, 5, 7, 8, 16] {
            let mut out_flash = vec![0.0f32; seq_len * head_dim];
            flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, bs);
            assert_close(&out_flash, &out_ref, 1e-4, &format!("flash_bs{bs}"));
        }
    }

    #[test]
    fn test_flash_seq1() {
        let (seq_len, head_dim) = (1, 8);
        let q = make_data(seq_len * head_dim, 440);
        let k = make_data(seq_len * head_dim, 441);
        let v = make_data(seq_len * head_dim, 442);
        let mut out = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out, seq_len, head_dim, 4);
        assert_close(&out, &v, 1e-6, "flash_seq1");
    }

    #[test]
    fn test_flash_numerical_stability() {
        let (seq_len, head_dim) = (4, 4);
        let q = vec![80.0f32; seq_len * head_dim];
        let k = vec![80.0f32; seq_len * head_dim];
        let v = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out, seq_len, head_dim, 2);
        for val in &out {
            assert!(val.is_finite(), "flash stability: non-finite {val}");
        }
    }

    #[test]
    fn test_flash_weights_sum_to_one() {
        let (seq_len, head_dim) = (6, 4);
        let q = make_data(seq_len * head_dim, 450);
        let k = make_data(seq_len * head_dim, 451);
        let v_ones = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v_ones, &mut out, seq_len, head_dim, 2);
        for (i, val) in out.iter().enumerate() {
            assert!((*val - 1.0).abs() < 1e-4, "flash weight sum [{i}]: {val} != 1.0");
        }
    }

    #[test]
    fn test_flash_non_power_of_2_dim() {
        let (seq_len, head_dim) = (5, 7);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 455);
        let k = make_data(seq_len * head_dim, 456);
        let v = make_data(seq_len * head_dim, 457);
        let mut out_flash = vec![0.0f32; seq_len * head_dim];
        let mut out_ref = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, 3);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_ref, seq_len, head_dim, scale);
        assert_close(&out_flash, &out_ref, 1e-4, "flash_dim7");
    }

    // ── 6. ALiBi attention tests ───────────────────────────────────────

    #[test]
    fn test_alibi_basic() {
        let (seq_len, head_dim) = (4, 8);
        let slope = 0.5;
        let q = make_data(seq_len * head_dim, 500);
        let k = make_data(seq_len * head_dim, 501);
        let v = make_data(seq_len * head_dim, 502);
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, slope);
        let expected = ref_alibi_f64(&q, &k, &v, seq_len, head_dim, slope);
        assert_close(&out, &expected, 1e-4, "alibi_basic");
    }

    #[test]
    fn test_alibi_zero_slope_is_sdpa() {
        let (seq_len, head_dim) = (4, 4);
        let q = make_data(seq_len * head_dim, 510);
        let k = make_data(seq_len * head_dim, 511);
        let v = make_data(seq_len * head_dim, 512);
        let mut out_alibi = vec![0.0f32; seq_len * head_dim];
        let mut out_sdpa = vec![0.0f32; seq_len * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        attention_with_alibi_f32(&q, &k, &v, &mut out_alibi, seq_len, head_dim, 0.0);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_sdpa, seq_len, head_dim, scale);
        assert_close(&out_alibi, &out_sdpa, 1e-5, "alibi_zero_slope");
    }

    #[test]
    fn test_alibi_distance_penalty() {
        // With high slope, distant positions are suppressed
        let (seq_len, head_dim) = (8, 4);
        let q = vec![0.0f32; seq_len * head_dim];
        let k = vec![0.0f32; seq_len * head_dim];
        let v_ones = vec![1.0f32; seq_len * head_dim];
        // With zero QK and slope, bias = -slope * |i-j|
        // Higher slope means more local attention
        let mut out_low = vec![0.0f32; seq_len * head_dim];
        let mut out_high = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v_ones, &mut out_low, seq_len, head_dim, 0.01);
        attention_with_alibi_f32(&q, &k, &v_ones, &mut out_high, seq_len, head_dim, 10.0);
        // Both should produce finite results with sum=1 (since V=1)
        for val in out_low.iter().chain(out_high.iter()) {
            assert!(val.is_finite(), "alibi_distance: non-finite");
        }
    }

    #[test]
    fn test_alibi_seq1() {
        let (seq_len, head_dim) = (1, 8);
        let slope = 1.0;
        let q = make_data(seq_len * head_dim, 520);
        let k = make_data(seq_len * head_dim, 521);
        let v = make_data(seq_len * head_dim, 522);
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, slope);
        assert_close(&out, &v, 1e-6, "alibi_seq1");
    }

    #[test]
    fn test_alibi_head_dim_1() {
        let (seq_len, head_dim) = (4, 1);
        let slope = 0.5;
        let q = make_data(seq_len * head_dim, 525);
        let k = make_data(seq_len * head_dim, 526);
        let v = make_data(seq_len * head_dim, 527);
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, slope);
        let expected = ref_alibi_f64(&q, &k, &v, seq_len, head_dim, slope);
        assert_close(&out, &expected, 1e-4, "alibi_hd1");
    }

    #[test]
    fn test_alibi_numerical_stability() {
        let (seq_len, head_dim) = (4, 4);
        let q = vec![50.0f32; seq_len * head_dim];
        let k = vec![50.0f32; seq_len * head_dim];
        let v = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, 1.0);
        for val in &out {
            assert!(val.is_finite(), "alibi stability: non-finite {val}");
        }
    }

    #[test]
    fn test_alibi_weights_sum_to_one() {
        let (seq_len, head_dim) = (5, 4);
        let slope = 0.3;
        let q = make_data(seq_len * head_dim, 530);
        let k = make_data(seq_len * head_dim, 531);
        let v_ones = vec![1.0f32; seq_len * head_dim];
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v_ones, &mut out, seq_len, head_dim, slope);
        for val in &out {
            assert!((*val - 1.0).abs() < 1e-5, "alibi weight sum: {val} != 1.0");
        }
    }

    #[test]
    fn test_alibi_various_slopes() {
        let (seq_len, head_dim) = (4, 4);
        let q = make_data(seq_len * head_dim, 540);
        let k = make_data(seq_len * head_dim, 541);
        let v = make_data(seq_len * head_dim, 542);
        for &slope in &[0.01, 0.1, 0.5, 1.0, 2.0] {
            let mut out = vec![0.0f32; seq_len * head_dim];
            attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, slope);
            let expected = ref_alibi_f64(&q, &k, &v, seq_len, head_dim, slope);
            assert_close(&out, &expected, 1e-4, &format!("alibi_slope_{slope}"));
        }
    }

    // ── Cross-function tests ───────────────────────────────────────────

    #[test]
    fn test_causal_last_row_equals_full() {
        // The last row in causal attention attends to all positions
        let (seq_len, head_dim) = (6, 8);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 600);
        let k = make_data(seq_len * head_dim, 601);
        let v = make_data(seq_len * head_dim, 602);
        let mut out_c = vec![0.0f32; seq_len * head_dim];
        let mut out_f = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out_c, seq_len, head_dim, scale);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_f, seq_len, head_dim, scale);
        let last = (seq_len - 1) * head_dim;
        assert_close(
            &out_c[last..last + head_dim],
            &out_f[last..last + head_dim],
            1e-5,
            "causal_last_eq_full",
        );
    }

    #[test]
    fn test_gqa_different_group_sizes() {
        for (nq, nkv) in [(2, 1), (4, 2), (6, 2), (8, 4), (8, 1)] {
            let (seq_len, head_dim) = (3, 4);
            let q = make_data(nq * seq_len * head_dim, 610 + nq as u32);
            let k = make_data(nkv * seq_len * head_dim, 620 + nkv as u32);
            let v = make_data(nkv * seq_len * head_dim, 630 + nkv as u32);
            let mut out = vec![0.0f32; nq * seq_len * head_dim];
            grouped_query_attention_f32(&q, &k, &v, &mut out, nq, nkv, seq_len, head_dim);
            for val in &out {
                assert!(val.is_finite(), "gqa({nq},{nkv}): non-finite");
            }
        }
    }

    #[test]
    fn test_flash_exact_block_multiple() {
        // seq_len is exact multiple of block_size
        let (seq_len, head_dim, bs) = (8, 4, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 650);
        let k = make_data(seq_len * head_dim, 651);
        let v = make_data(seq_len * head_dim, 652);
        let mut out_flash = vec![0.0f32; seq_len * head_dim];
        let mut out_ref = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, bs);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_ref, seq_len, head_dim, scale);
        assert_close(&out_flash, &out_ref, 1e-4, "flash_exact_mult");
    }

    #[test]
    fn test_sdpa_larger_sequence() {
        let (seq_len, head_dim) = (16, 32);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 700);
        let k = make_data(seq_len * head_dim, 701);
        let v = make_data(seq_len * head_dim, 702);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-4, "sdpa_large_seq");
    }

    #[test]
    fn test_causal_larger_sequence() {
        let (seq_len, head_dim) = (16, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 710);
        let k = make_data(seq_len * head_dim, 711);
        let v = make_data(seq_len * head_dim, 712);
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_causal_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-4, "causal_large_seq");
    }

    #[test]
    fn test_alibi_symmetry_at_diagonal() {
        // At position i, attending to itself has distance 0, so no penalty
        let (seq_len, head_dim) = (4, 4);
        let slope = 1.0;
        // Use identity-like Q and K so self-attention dominates
        let mut q = vec![0.0f32; seq_len * head_dim];
        let mut k = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            q[i * head_dim + (i % head_dim)] = 10.0;
            k[i * head_dim + (i % head_dim)] = 10.0;
        }
        let v = make_data(seq_len * head_dim, 720);
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, slope);
        for val in &out {
            assert!(val.is_finite(), "alibi_symmetry: non-finite");
        }
    }

    #[test]
    fn test_mha_correctness_vs_ref() {
        let (num_heads, seq_len, head_dim) = (4, 4, 8);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 800);
        let k = make_data(total, 801);
        let v = make_data(total, 802);
        let mut out = vec![0.0f32; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, num_heads, seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let hs = seq_len * head_dim;
        for h in 0..num_heads {
            let expected = ref_sdpa_f64(
                &q[h * hs..(h + 1) * hs],
                &k[h * hs..(h + 1) * hs],
                &v[h * hs..(h + 1) * hs],
                seq_len,
                head_dim,
                scale,
            );
            assert_close(&out[h * hs..(h + 1) * hs], &expected, 1e-5, &format!("mha_ref_head{h}"));
        }
    }

    #[test]
    fn test_sdpa_zero_scale() {
        // scale=0 → all scores are 0 → uniform attention
        let (seq_len, head_dim) = (3, 4);
        let q = make_data(seq_len * head_dim, 810);
        let k = make_data(seq_len * head_dim, 811);
        let v = make_data(seq_len * head_dim, 812);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, 0.0);
        // Uniform attention → output is mean of all V rows for each row
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mean: f32 =
                    (0..seq_len).map(|j| v[j * head_dim + d]).sum::<f32>() / seq_len as f32;
                assert!(
                    (out[i * head_dim + d] - mean).abs() < 1e-5,
                    "zero_scale[{i}][{d}]: {} vs {}",
                    out[i * head_dim + d],
                    mean,
                );
            }
        }
    }

    #[test]
    fn test_flash_large_dim() {
        let (seq_len, head_dim) = (8, 64);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 820);
        let k = make_data(seq_len * head_dim, 821);
        let v = make_data(seq_len * head_dim, 822);
        let mut out_flash = vec![0.0f32; seq_len * head_dim];
        let mut out_ref = vec![0.0f32; seq_len * head_dim];
        flash_attention_tiled_f32(&q, &k, &v, &mut out_flash, seq_len, head_dim, 4);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_ref, seq_len, head_dim, scale);
        assert_close(&out_flash, &out_ref, 1e-3, "flash_large_dim");
    }

    #[test]
    fn test_softmax_helper_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        scalar_softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum: {sum}");
        // Should be monotonically increasing
        for w in data.windows(2) {
            assert!(w[0] <= w[1], "softmax not monotonic");
        }
    }

    #[test]
    fn test_softmax_helper_single() {
        let mut data = vec![42.0];
        scalar_softmax_inplace(&mut data);
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_helper_empty() {
        let mut data: Vec<f32> = vec![];
        scalar_softmax_inplace(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_softmax_helper_large_values() {
        let mut data = vec![1000.0, 1001.0, 1002.0];
        scalar_softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax large: sum={sum}");
        for val in &data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_softmax_helper_negative_values() {
        let mut data = vec![-100.0, -200.0, -300.0];
        scalar_softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax neg: sum={sum}");
    }

    #[test]
    fn test_gqa_weights_sum_to_one() {
        let (num_q, num_kv, seq_len, head_dim) = (4, 2, 4, 4);
        let q = make_data(num_q * seq_len * head_dim, 900);
        let k = make_data(num_kv * seq_len * head_dim, 901);
        let v_ones = vec![1.0f32; num_kv * seq_len * head_dim];
        let mut out = vec![0.0f32; num_q * seq_len * head_dim];
        grouped_query_attention_f32(&q, &k, &v_ones, &mut out, num_q, num_kv, seq_len, head_dim);
        for val in &out {
            assert!((*val - 1.0).abs() < 1e-5, "gqa weight sum: {val}");
        }
    }

    #[test]
    fn test_mha_weights_sum_to_one() {
        let (num_heads, seq_len, head_dim) = (3, 4, 4);
        let total = num_heads * seq_len * head_dim;
        let q = make_data(total, 910);
        let k = make_data(total, 911);
        let v_ones = vec![1.0f32; total];
        let mut out = vec![0.0f32; total];
        multi_head_attention_f32(&q, &k, &v_ones, &mut out, num_heads, seq_len, head_dim);
        for val in &out {
            assert!((*val - 1.0).abs() < 1e-5, "mha weight sum: {val}");
        }
    }

    #[test]
    fn test_sdpa_dim_3_tail_handling() {
        // head_dim=3 forces NEON tail element handling (3 % 4 = 3)
        let (seq_len, head_dim) = (4, 3);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 920);
        let k = make_data(seq_len * head_dim, 921);
        let v = make_data(seq_len * head_dim, 922);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "sdpa_dim3");
    }

    #[test]
    fn test_sdpa_dim_5_tail_handling() {
        // head_dim=5 forces NEON tail handling (5 % 4 = 1)
        let (seq_len, head_dim) = (4, 5);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 930);
        let k = make_data(seq_len * head_dim, 931);
        let v = make_data(seq_len * head_dim, 932);
        let mut out = vec![0.0f32; seq_len * head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = ref_sdpa_f64(&q, &k, &v, seq_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "sdpa_dim5");
    }

    #[test]
    fn test_causal_monotonic_context() {
        // Each successive row attends to strictly more positions
        // So output variance should generally differ between rows
        let (seq_len, head_dim) = (4, 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q = make_data(seq_len * head_dim, 940);
        let k = make_data(seq_len * head_dim, 941);
        let v = make_data(seq_len * head_dim, 942);
        let mut out = vec![0.0f32; seq_len * head_dim];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        // Just verify non-degenerate output (each row finite)
        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!(out[i * head_dim + d].is_finite());
            }
        }
    }

    #[test]
    fn test_alibi_large_slope_makes_local() {
        // Very large slope should make attention almost entirely local (self-attend)
        let (seq_len, head_dim) = (8, 4);
        let slope = 100.0;
        let mut q = vec![0.0f32; seq_len * head_dim];
        let mut k = vec![0.0f32; seq_len * head_dim];
        // Make each position unique
        for i in 0..seq_len {
            for d in 0..head_dim {
                q[i * head_dim + d] = (i * head_dim + d) as f32 * 0.01;
                k[i * head_dim + d] = (i * head_dim + d) as f32 * 0.01;
            }
        }
        let mut v = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            v[i * head_dim] = i as f32;
        }
        let mut out = vec![0.0f32; seq_len * head_dim];
        attention_with_alibi_f32(&q, &k, &v, &mut out, seq_len, head_dim, slope);
        // With huge slope, attention should be nearly one-hot on self
        for i in 0..seq_len {
            let val = out[i * head_dim];
            // Should be close to v[i][0] = i
            assert!((val - i as f32).abs() < 0.5, "alibi_local[{i}]: got {val}, expected ~{i}");
        }
    }
}
