//! ARM NEON optimized quantized attention kernel for 1-bit inference.
//!
//! Implements efficient scaled dot-product attention that operates directly
//! on 2-bit ternary-packed weights (I2_S format) without full
//! dequantization, using ARM NEON SIMD intrinsics on Apple Silicon / AArch64.
//!
//! I2_S encoding (2 bits per value, 4 values per byte, LSB-first):
//! - `0b00` → 0
//! - `0b01` → +1
//! - `0b11` → −1
//! - `0b10` → unused (treated as 0)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── I2_S decode helpers ────────────────────────────────────────────────

/// Decode a single 2-bit I2_S code to its signed integer value.
#[inline(always)]
fn decode_i2s(bits: u8) -> i8 {
    match bits & 0x03 {
        0b01 => 1,
        0b11 => -1,
        _ => 0, // 0b00 = 0, 0b10 = unused → 0
    }
}

/// Decode a single 2-bit I2_S code to f32.
#[inline(always)]
fn decode_i2s_f32(bits: u8) -> f32 {
    match bits & 0x03 {
        0b01 => 1.0,
        0b11 => -1.0,
        _ => 0.0,
    }
}

/// Unpack one byte into 4 ternary i8 values (LSB-first).
#[inline(always)]
fn unpack_byte_i2s(byte: u8) -> [i8; 4] {
    [decode_i2s(byte), decode_i2s(byte >> 2), decode_i2s(byte >> 4), decode_i2s(byte >> 6)]
}

// ── Scalar fallback helpers ────────────────────────────────────────────

/// Scalar dot product of f32 query with packed I2_S values.
#[cfg(test)]
fn scalar_dot_i2s(query: &[f32], packed: &[u8], scale: f32) -> f32 {
    let mut acc = 0.0f32;
    let full_bytes = query.len() / 4;
    let remainder = query.len() % 4;

    for i in 0..full_bytes {
        let vals = unpack_byte_i2s(packed[i]);
        acc += query[i * 4] * vals[0] as f32;
        acc += query[i * 4 + 1] * vals[1] as f32;
        acc += query[i * 4 + 2] * vals[2] as f32;
        acc += query[i * 4 + 3] * vals[3] as f32;
    }

    if remainder > 0 {
        let byte = packed[full_bytes];
        for r in 0..remainder {
            let val = decode_i2s_f32(byte >> (r * 2));
            acc += query[full_bytes * 4 + r] * val;
        }
    }

    acc * scale
}

// ── NEON-accelerated kernels ───────────────────────────────────────────

/// Compute Q·K^T where K is 2-bit I2_S packed.
///
/// - `queries`: `[seq_len * head_dim]` flattened query vectors (f32)
/// - `keys_packed`: `[seq_len * packed_dim]` packed I2_S key vectors
///   where `packed_dim = ceil(head_dim / 4)`
/// - `key_scale`: dequantization scale for keys
/// - Returns: `[seq_len * seq_len]` attention logits
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_qk_dot_i2_f32(
    queries: &[f32],
    keys_packed: &[u8],
    key_scale: f32,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let packed_dim = (head_dim + 3) / 4;
    let mut output = vec![0.0f32; seq_len * seq_len];

    let scale_vec = vdupq_n_f32(key_scale);

    for q_idx in 0..seq_len {
        let q_offset = q_idx * head_dim;
        let q_slice = &queries[q_offset..q_offset + head_dim];

        for k_idx in 0..seq_len {
            let k_offset = k_idx * packed_dim;
            let k_slice = &keys_packed[k_offset..k_offset + packed_dim];

            let mut acc = vdupq_n_f32(0.0);
            let full_bytes = head_dim / 4;
            let remainder = head_dim % 4;

            // NEON: process 4 values (1 packed byte) at a time
            for i in 0..full_bytes {
                let byte = k_slice[i];
                let vals = unpack_byte_i2s(byte);
                let unpacked = [vals[0] as f32, vals[1] as f32, vals[2] as f32, vals[3] as f32];
                let k_vec = vld1q_f32(unpacked.as_ptr());
                let q_vec = vld1q_f32(q_slice.as_ptr().add(i * 4));
                acc = vfmaq_f32(acc, q_vec, k_vec);
            }

            // Horizontal sum
            let mut dot = vaddvq_f32(acc);

            // Scalar tail
            for r in 0..remainder {
                let byte = k_slice[full_bytes];
                let val = decode_i2s_f32(byte >> (r * 2));
                dot += q_slice[full_bytes * 4 + r] * val;
            }

            // Apply scale
            output[q_idx * seq_len + k_idx] = dot * key_scale;
        }
    }

    output
}

/// Scale attention scores by 1/√d_k and apply causal mask.
///
/// - `qk_scores`: `[seq_len * seq_len]` attention logits (modified in place)
/// - `seq_len`: sequence length
/// - `head_dim`: dimension per head (used to compute 1/√d_k)
///
/// Positions where `k_idx > q_idx` are set to `-f32::INFINITY` (causal mask).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_attention_score_f32(qk_scores: &mut [f32], seq_len: usize, head_dim: usize) {
    let inv_sqrt_dk = 1.0 / (head_dim as f32).sqrt();
    let scale_vec = vdupq_n_f32(inv_sqrt_dk);
    let neg_inf = f32::NEG_INFINITY;
    let neg_inf_vec = vdupq_n_f32(neg_inf);

    for q_idx in 0..seq_len {
        let row_offset = q_idx * seq_len;

        // Scale and mask the valid (causal) region [0..=q_idx]
        let valid_len = q_idx + 1;
        let chunks = valid_len / 4;
        let remainder = valid_len % 4;

        for c in 0..chunks {
            let base = row_offset + c * 4;
            let v = vld1q_f32(qk_scores.as_ptr().add(base));
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(qk_scores.as_mut_ptr().add(base), scaled);
        }

        let tail_start = chunks * 4;
        for r in 0..remainder {
            qk_scores[row_offset + tail_start + r] *= inv_sqrt_dk;
        }

        // Apply causal mask: set future positions to -inf
        let masked_start = q_idx + 1;
        let masked_len = seq_len - masked_start;
        let masked_chunks = masked_len / 4;
        let masked_remainder = masked_len % 4;

        for c in 0..masked_chunks {
            let base = row_offset + masked_start + c * 4;
            vst1q_f32(qk_scores.as_mut_ptr().add(base), neg_inf_vec);
        }

        let masked_tail_start = masked_start + masked_chunks * 4;
        for r in 0..masked_remainder {
            qk_scores[row_offset + masked_tail_start + r] = neg_inf;
        }
    }
}

/// Compute attention_weights · V where V is 2-bit I2_S packed.
///
/// - `attention_weights`: `[seq_len * seq_len]` softmaxed attention weights
/// - `values_packed`: `[seq_len * packed_dim]` packed I2_S value vectors
/// - `value_scale`: dequantization scale for values
/// - Returns: `[seq_len * head_dim]` output vectors
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_weighted_sum_i2_f32(
    attention_weights: &[f32],
    values_packed: &[u8],
    value_scale: f32,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let packed_dim = (head_dim + 3) / 4;
    let mut output = vec![0.0f32; seq_len * head_dim];

    for q_idx in 0..seq_len {
        let weight_row = &attention_weights[q_idx * seq_len..(q_idx + 1) * seq_len];

        // For each output dimension, accumulate weighted sum
        let full_bytes = head_dim / 4;
        let remainder = head_dim % 4;

        // Accumulate across all value positions
        for v_idx in 0..seq_len {
            let w = weight_row[v_idx];
            if w == 0.0 {
                continue;
            }
            let w_vec = vdupq_n_f32(w);
            let v_offset = v_idx * packed_dim;
            let v_slice = &values_packed[v_offset..v_offset + packed_dim];

            // NEON: process 4 elements at a time
            for i in 0..full_bytes {
                let byte = v_slice[i];
                let vals = unpack_byte_i2s(byte);
                let unpacked = [vals[0] as f32, vals[1] as f32, vals[2] as f32, vals[3] as f32];
                let v_vec = vld1q_f32(unpacked.as_ptr());
                let out_base = q_idx * head_dim + i * 4;
                let existing = vld1q_f32(output.as_ptr().add(out_base));
                let result = vfmaq_f32(existing, w_vec, v_vec);
                vst1q_f32(output.as_mut_ptr().add(out_base), result);
            }

            // Scalar tail
            for r in 0..remainder {
                let byte = v_slice[full_bytes];
                let val = decode_i2s_f32(byte >> (r * 2));
                output[q_idx * head_dim + full_bytes * 4 + r] += w * val;
            }
        }

        // Apply value scale to this row
        let scale_vec = vdupq_n_f32(value_scale);
        let row_start = q_idx * head_dim;
        let row_chunks = head_dim / 4;
        let row_remainder = head_dim % 4;

        for c in 0..row_chunks {
            let base = row_start + c * 4;
            let v = vld1q_f32(output.as_ptr().add(base));
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(output.as_mut_ptr().add(base), scaled);
        }

        let tail = row_start + row_chunks * 4;
        for r in 0..row_remainder {
            output[tail + r] *= value_scale;
        }
    }

    output
}

/// Softmax over a row of f32 values (NEON-accelerated).
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_row(row: &mut [f32]) {
    let len = row.len();
    if len == 0 {
        return;
    }

    // Find max for numerical stability
    let chunks = len / 4;
    let remainder = len % 4;
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);

    for c in 0..chunks {
        let v = vld1q_f32(row.as_ptr().add(c * 4));
        max_vec = vmaxq_f32(max_vec, v);
    }
    let mut max_val = vmaxvq_f32(max_vec);
    for r in 0..remainder {
        let val = row[chunks * 4 + r];
        if val > max_val {
            max_val = val;
        }
    }

    // Compute exp(x - max) and sum
    let max_broadcast = vdupq_n_f32(max_val);
    let mut sum = 0.0f32;

    for c in 0..chunks {
        let base = c * 4;
        let v = vld1q_f32(row.as_ptr().add(base));
        let shifted = vsubq_f32(v, max_broadcast);
        // exp via scalar (NEON has no direct exp)
        let mut exp_vals = [0.0f32; 4];
        vst1q_f32(exp_vals.as_mut_ptr(), shifted);
        for val in &mut exp_vals {
            *val = val.exp();
        }
        let exp_vec = vld1q_f32(exp_vals.as_ptr());
        vst1q_f32(row.as_mut_ptr().add(base), exp_vec);
        sum += vaddvq_f32(exp_vec);
    }

    let tail_start = chunks * 4;
    for r in 0..remainder {
        let val = (row[tail_start + r] - max_val).exp();
        row[tail_start + r] = val;
        sum += val;
    }

    // Normalize
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = vdupq_n_f32(inv_sum);

        for c in 0..chunks {
            let base = c * 4;
            let v = vld1q_f32(row.as_ptr().add(base));
            let normed = vmulq_f32(v, inv_sum_vec);
            vst1q_f32(row.as_mut_ptr().add(base), normed);
        }

        for r in 0..remainder {
            row[tail_start + r] *= inv_sum;
        }
    }
}

/// Full multi-head attention with quantized KV in I2_S format.
///
/// - `queries`: `[num_heads * seq_len * head_dim]` flattened (f32)
/// - `keys_packed`: `[num_heads * seq_len * packed_dim]` packed I2_S keys
/// - `values_packed`: `[num_heads * seq_len * packed_dim]` packed I2_S values
/// - `key_scale`, `value_scale`: dequantization scales
/// - Returns: `[num_heads * seq_len * head_dim]` output
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_multi_head_attention_i2(
    queries: &[f32],
    keys_packed: &[u8],
    values_packed: &[u8],
    key_scale: f32,
    value_scale: f32,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let packed_dim = (head_dim + 3) / 4;
    let q_head_stride = seq_len * head_dim;
    let kv_head_stride = seq_len * packed_dim;
    let out_head_stride = seq_len * head_dim;

    let mut output = vec![0.0f32; num_heads * seq_len * head_dim];

    for h in 0..num_heads {
        let q_start = h * q_head_stride;
        let k_start = h * kv_head_stride;
        let v_start = h * kv_head_stride;

        let head_queries = &queries[q_start..q_start + q_head_stride];
        let head_keys = &keys_packed[k_start..k_start + kv_head_stride];
        let head_values = &values_packed[v_start..v_start + kv_head_stride];

        // Q·K^T
        let mut qk = neon_qk_dot_i2_f32(head_queries, head_keys, key_scale, seq_len, head_dim);

        // Scale and causal mask
        neon_attention_score_f32(&mut qk, seq_len, head_dim);

        // Softmax per row
        for q_idx in 0..seq_len {
            let row_start = q_idx * seq_len;
            let row = &mut qk[row_start..row_start + seq_len];
            neon_softmax_row(row);
        }

        // Weighted sum with values
        let head_out = neon_weighted_sum_i2_f32(&qk, head_values, value_scale, seq_len, head_dim);

        let out_start = h * out_head_stride;
        output[out_start..out_start + out_head_stride].copy_from_slice(&head_out);
    }

    output
}

/// Append new quantized KV entries to a packed I2_S cache buffer.
///
/// - `cache`: pre-allocated packed I2_S cache buffer
/// - `new_kv`: packed I2_S data to append
/// - `cache_len`: current number of valid bytes; updated after append
/// - `max_len`: maximum capacity in bytes
///
/// If appending would exceed `max_len`, only appends up to the boundary.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_append_i2(
    cache: &mut Vec<u8>,
    new_kv: &[u8],
    cache_len: &mut usize,
    max_len: usize,
) {
    let available = max_len.saturating_sub(*cache_len);
    let copy_len = new_kv.len().min(available);

    if copy_len == 0 {
        return;
    }

    // Ensure capacity
    let needed = *cache_len + copy_len;
    if cache.len() < needed {
        cache.resize(needed, 0);
    }

    // NEON-accelerated copy: 16 bytes at a time
    let chunks = copy_len / 16;
    let remainder = copy_len % 16;

    for c in 0..chunks {
        let src_base = c * 16;
        let dst_base = *cache_len + c * 16;
        let v = vld1q_u8(new_kv.as_ptr().add(src_base));
        vst1q_u8(cache.as_mut_ptr().add(dst_base), v);
    }

    // Scalar tail
    let tail_start = chunks * 16;
    for r in 0..remainder {
        cache[*cache_len + tail_start + r] = new_kv[tail_start + r];
    }

    *cache_len += copy_len;
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── I2_S packing helpers for tests ─────────────────────────────────

    /// Pack 4 ternary values (-1, 0, +1) into a single byte.
    fn pack_i2s(v0: i8, v1: i8, v2: i8, v3: i8) -> u8 {
        fn encode(v: i8) -> u8 {
            match v {
                1 => 0b01,
                -1 => 0b11,
                _ => 0b00,
            }
        }
        encode(v0) | (encode(v1) << 2) | (encode(v2) << 4) | (encode(v3) << 6)
    }

    /// Pack a slice of ternary values into I2_S bytes.
    fn pack_slice(vals: &[i8]) -> Vec<u8> {
        let mut packed = Vec::new();
        for chunk in vals.chunks(4) {
            let v0 = chunk[0];
            let v1 = if chunk.len() > 1 { chunk[1] } else { 0 };
            let v2 = if chunk.len() > 2 { chunk[2] } else { 0 };
            let v3 = if chunk.len() > 3 { chunk[3] } else { 0 };
            packed.push(pack_i2s(v0, v1, v2, v3));
        }
        packed
    }

    /// Scalar reference softmax for validation.
    fn reference_softmax(row: &mut [f32]) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Scalar reference for Q·K^T dot with I2_S packed keys.
    fn reference_qk_dot(
        queries: &[f32],
        keys_packed: &[u8],
        key_scale: f32,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let packed_dim = (head_dim + 3) / 4;
        let mut output = vec![0.0f32; seq_len * seq_len];
        for q in 0..seq_len {
            for k in 0..seq_len {
                output[q * seq_len + k] = scalar_dot_i2s(
                    &queries[q * head_dim..(q + 1) * head_dim],
                    &keys_packed[k * packed_dim..(k + 1) * packed_dim],
                    key_scale,
                );
            }
        }
        output
    }

    // ── qk_dot tests ──────────────────────────────────────────────────

    #[test]
    fn qk_dot_single_position() {
        let head_dim = 4;
        let seq_len = 1;
        let queries = vec![1.0, 2.0, 3.0, 4.0];
        let keys = pack_slice(&[1, 0, -1, 1]);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, seq_len, head_dim) };
        // 1*1 + 2*0 + 3*(-1) + 4*1 = 1 - 3 + 4 = 2.0
        assert!((result[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_multi_position() {
        let head_dim = 4;
        let seq_len = 2;
        let queries = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let keys_vals: Vec<i8> = vec![1, 1, 0, 0, 0, -1, 1, 0];
        let keys = pack_slice(&keys_vals);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, seq_len, head_dim) };
        // q0·k0 = 1*1 = 1, q0·k1 = 1*0 = 0
        // q1·k0 = 1*1 = 1, q1·k1 = 1*(-1) = -1
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 1.0).abs() < 1e-5);
        assert!((result[3] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_all_zeros() {
        let head_dim = 8;
        let seq_len = 2;
        let queries = vec![1.0; seq_len * head_dim];
        let keys = pack_slice(&vec![0i8; seq_len * head_dim]);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, seq_len, head_dim) };
        for v in &result {
            assert!(*v == 0.0, "expected 0.0, got {v}");
        }
    }

    #[test]
    fn qk_dot_all_ones() {
        let head_dim = 8;
        let seq_len = 1;
        let queries = vec![1.0; head_dim];
        let keys = pack_slice(&vec![1i8; head_dim]);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, seq_len, head_dim) };
        assert!((result[0] - head_dim as f32).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_all_neg_ones() {
        let head_dim = 8;
        let seq_len = 1;
        let queries = vec![1.0; head_dim];
        let keys = pack_slice(&vec![-1i8; head_dim]);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, seq_len, head_dim) };
        assert!((result[0] - (-(head_dim as f32))).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_mixed_values() {
        let head_dim = 4;
        let seq_len = 1;
        let queries = vec![2.0, 3.0, 4.0, 5.0];
        let keys = pack_slice(&[1, -1, 0, 1]);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, seq_len, head_dim) };
        // 2*1 + 3*(-1) + 4*0 + 5*1 = 2 - 3 + 0 + 5 = 4.0
        assert!((result[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_scale_factor() {
        let head_dim = 4;
        let seq_len = 1;
        let queries = vec![1.0; head_dim];
        let keys = pack_slice(&[1, 1, 1, 1]);
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 0.5, seq_len, head_dim) };
        // (1+1+1+1) * 0.5 = 2.0
        assert!((result[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_head_dim_4() {
        let hd = 4;
        let sl = 2;
        let q = vec![1.0, -1.0, 1.0, -1.0, 0.5, 0.5, 0.5, 0.5];
        let k = pack_slice(&[1, 1, -1, -1, 0, 1, 0, -1]);
        let expected = reference_qk_dot(&q, &k, 1.0, sl, hd);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, 1.0, sl, hd) };
        for (i, (&e, &r)) in expected.iter().zip(result.iter()).enumerate() {
            assert!((e - r).abs() < 1e-5, "mismatch at {i}: {e} vs {r}");
        }
    }

    #[test]
    fn qk_dot_head_dim_8() {
        let hd = 8;
        let sl = 1;
        let q = vec![1.0; hd];
        let k = pack_slice(&[1, -1, 1, -1, 1, -1, 1, -1]);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, 1.0, sl, hd) };
        // sum of alternating +1/-1 = 0
        assert!((result[0]).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_head_dim_16() {
        let hd = 16;
        let sl = 1;
        let q = vec![0.25; hd];
        let k = pack_slice(&vec![1i8; hd]);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, 1.0, sl, hd) };
        // 16 * 0.25 * 1 = 4.0
        assert!((result[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_head_dim_32() {
        let hd = 32;
        let sl = 1;
        let q = vec![1.0; hd];
        let k = pack_slice(&vec![1i8; hd]);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, 1.0, sl, hd) };
        assert!((result[0] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn qk_dot_head_dim_64() {
        let hd = 64;
        let sl = 1;
        let q = vec![0.5; hd];
        let k_vals: Vec<i8> = (0..hd).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let k = pack_slice(&k_vals);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, 2.0, sl, hd) };
        // each pair: 0.5*1 + 0.5*(-1) = 0, total = 0, * 2.0 = 0.0
        assert!((result[0]).abs() < 1e-5);
    }

    // ── attention_score tests ──────────────────────────────────────────

    #[test]
    fn attention_score_identity() {
        let seq_len = 1;
        let head_dim = 4;
        let mut scores = vec![2.0];
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        let expected = 2.0 / (4.0f32).sqrt();
        assert!((scores[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn attention_score_uniform() {
        let seq_len = 2;
        let head_dim = 4;
        let mut scores = vec![1.0; seq_len * seq_len];
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        let scale = 1.0 / (4.0f32).sqrt();
        assert!((scores[0] - scale).abs() < 1e-5);
        assert!(scores[1] == f32::NEG_INFINITY); // causal mask
        assert!((scores[2] - scale).abs() < 1e-5);
        assert!((scores[3] - scale).abs() < 1e-5);
    }

    #[test]
    fn attention_score_peaked() {
        let seq_len = 3;
        let head_dim = 16;
        let mut scores = vec![0.0; seq_len * seq_len];
        scores[0] = 10.0; // q0 attends strongly to k0
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        let scale = 1.0 / (16.0f32).sqrt();
        assert!((scores[0] - 10.0 * scale).abs() < 1e-5);
    }

    #[test]
    fn attention_score_scale_correctness() {
        let head_dim = 64;
        let seq_len = 1;
        let mut scores = vec![8.0];
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        let expected = 8.0 / (64.0f32).sqrt();
        assert!((scores[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn attention_score_causal_mask() {
        let seq_len = 4;
        let head_dim = 4;
        let mut scores = vec![1.0; seq_len * seq_len];
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        // Row 0: only [0] is valid, [1..3] = -inf
        assert!(scores[1] == f32::NEG_INFINITY);
        assert!(scores[2] == f32::NEG_INFINITY);
        assert!(scores[3] == f32::NEG_INFINITY);
        // Row 1: [0,1] valid, [2,3] = -inf
        assert!(scores[6] == f32::NEG_INFINITY);
        assert!(scores[7] == f32::NEG_INFINITY);
        // Row 3: all valid (last row)
        let scale = 1.0 / 2.0;
        for j in 0..4 {
            assert!((scores[12 + j] - scale).abs() < 1e-5);
        }
    }

    #[test]
    fn attention_score_numerical_stability() {
        let seq_len = 1;
        let head_dim = 4;
        let mut scores = vec![1e6];
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        assert!(scores[0].is_finite());
    }

    #[test]
    fn attention_score_large_seq() {
        let seq_len = 32;
        let head_dim = 16;
        let mut scores = vec![1.0; seq_len * seq_len];
        unsafe { neon_attention_score_f32(&mut scores, seq_len, head_dim) };
        let scale = 1.0 / (16.0f32).sqrt();
        // Check diagonal is scaled
        for i in 0..seq_len {
            assert!((scores[i * seq_len + i] - scale).abs() < 1e-5);
        }
        // Check upper triangle is -inf
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                assert!(scores[i * seq_len + j] == f32::NEG_INFINITY);
            }
        }
    }

    // ── weighted_sum tests ─────────────────────────────────────────────

    #[test]
    fn weighted_sum_identity_weights() {
        let seq_len = 2;
        let head_dim = 4;
        // Identity-like: first row attends only to pos 0
        let mut weights = vec![0.0; seq_len * seq_len];
        weights[0] = 1.0; // q0 → v0
        weights[3] = 1.0; // q1 → v1
        let vals: Vec<i8> = vec![1, 0, -1, 1, -1, 1, 0, 0];
        let values = pack_slice(&vals);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        // q0 gets v0 = [1, 0, -1, 1]
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - (-1.0)).abs() < 1e-5);
        assert!((result[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_sum_uniform_weights() {
        let seq_len = 2;
        let head_dim = 4;
        let weights = vec![0.5; seq_len * seq_len];
        let vals: Vec<i8> = vec![1, 1, 1, 1, -1, -1, -1, -1];
        let values = pack_slice(&vals);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        // Both rows: avg of [1,1,1,1] and [-1,-1,-1,-1] = [0,0,0,0]
        for v in &result {
            assert!(v.abs() < 1e-5, "expected ~0, got {v}");
        }
    }

    #[test]
    fn weighted_sum_single_position() {
        let seq_len = 1;
        let head_dim = 4;
        let weights = vec![1.0];
        let values = pack_slice(&[1, -1, 0, 1]);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 2.0, seq_len, head_dim) };
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - (-2.0)).abs() < 1e-5);
        assert!((result[2] - 0.0).abs() < 1e-5);
        assert!((result[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_sum_all_zeros_values() {
        let seq_len = 2;
        let head_dim = 8;
        let weights = vec![0.5; seq_len * seq_len];
        let values = pack_slice(&vec![0i8; seq_len * head_dim]);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        for v in &result {
            assert!(*v == 0.0);
        }
    }

    #[test]
    fn weighted_sum_mixed() {
        let seq_len = 2;
        let head_dim = 4;
        let weights = vec![0.7, 0.3, 0.2, 0.8];
        let vals: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0];
        let values = pack_slice(&vals);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        // q0: 0.7*[1,0,0,0] + 0.3*[0,1,0,0] = [0.7, 0.3, 0, 0]
        assert!((result[0] - 0.7).abs() < 1e-5);
        assert!((result[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn weighted_sum_head_dim_8() {
        let seq_len = 1;
        let head_dim = 8;
        let weights = vec![1.0];
        let vals: Vec<i8> = vec![1, 1, -1, -1, 1, 1, -1, -1];
        let values = pack_slice(&vals);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 0.5, seq_len, head_dim) };
        let expected = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5];
        for (i, (&e, &r)) in expected.iter().zip(result.iter()).enumerate() {
            assert!((e - r).abs() < 1e-5, "dim {i}: {e} vs {r}");
        }
    }

    #[test]
    fn weighted_sum_head_dim_16() {
        let seq_len = 1;
        let head_dim = 16;
        let weights = vec![1.0];
        let values = pack_slice(&vec![1i8; head_dim]);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        for v in &result {
            assert!((*v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn weighted_sum_head_dim_32() {
        let seq_len = 1;
        let head_dim = 32;
        let weights = vec![1.0];
        let values = pack_slice(&vec![-1i8; head_dim]);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        for v in &result {
            assert!((*v - (-1.0)).abs() < 1e-5);
        }
    }

    // ── multi_head tests ───────────────────────────────────────────────

    #[test]
    fn multi_head_single_head() {
        let num_heads = 1;
        let seq_len = 1;
        let head_dim = 4;
        let queries = vec![1.0; head_dim];
        let keys = pack_slice(&[1, 0, 0, 0]);
        let values = pack_slice(&[0, 1, 0, 0]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(result.len(), head_dim);
        // After softmax on single element: weight=1.0, so output = value * scale
        assert!((result[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn multi_head_two_heads() {
        let num_heads = 2;
        let seq_len = 1;
        let head_dim = 4;
        let queries = vec![1.0; num_heads * seq_len * head_dim];
        let packed_dim = (head_dim + 3) / 4;
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(result.len(), num_heads * seq_len * head_dim);
    }

    #[test]
    fn multi_head_four_heads() {
        let num_heads = 4;
        let seq_len = 1;
        let head_dim = 4;
        let queries = vec![1.0; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(result.len(), num_heads * seq_len * head_dim);
    }

    #[test]
    fn multi_head_eight_heads() {
        let num_heads = 8;
        let seq_len = 2;
        let head_dim = 4;
        let queries = vec![0.5; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(result.len(), num_heads * seq_len * head_dim);
    }

    #[test]
    fn multi_head_output_shape() {
        let num_heads = 3;
        let seq_len = 4;
        let head_dim = 8;
        let queries = vec![1.0; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![0i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![0i8; num_heads * seq_len * head_dim]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(result.len(), num_heads * seq_len * head_dim);
    }

    #[test]
    fn multi_head_causal() {
        let num_heads = 1;
        let seq_len = 3;
        let head_dim = 4;
        let queries = vec![1.0; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values_raw: Vec<i8> = vec![
            1, 0, 0, 0, // v0
            0, 1, 0, 0, // v1
            0, 0, 1, 0, // v2
        ];
        let values = pack_slice(&values_raw);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        // Position 0 can only attend to position 0 → output ≈ v0
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1]).abs() < 1e-4);
    }

    #[test]
    fn multi_head_numerical_range() {
        let num_heads = 2;
        let seq_len = 2;
        let head_dim = 4;
        let queries = vec![10.0; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        for v in &result {
            assert!(v.is_finite(), "non-finite value: {v}");
        }
    }

    #[test]
    fn multi_head_deterministic() {
        let num_heads = 2;
        let seq_len = 3;
        let head_dim = 8;
        let queries = vec![0.1; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![-1i8; num_heads * seq_len * head_dim]);
        let r1 = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        let r2 = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(r1, r2);
    }

    // ── kv_cache tests ─────────────────────────────────────────────────

    #[test]
    fn kv_cache_empty_append() {
        let mut cache = Vec::new();
        let mut cache_len = 0usize;
        let new_data = vec![0xAB, 0xCD];
        unsafe { neon_kv_cache_append_i2(&mut cache, &new_data, &mut cache_len, 100) };
        assert_eq!(cache_len, 2);
        assert_eq!(cache[0], 0xAB);
        assert_eq!(cache[1], 0xCD);
    }

    #[test]
    fn kv_cache_single_append() {
        let mut cache = vec![0x11; 4];
        let mut cache_len = 4usize;
        let new_data = vec![0x22; 2];
        unsafe { neon_kv_cache_append_i2(&mut cache, &new_data, &mut cache_len, 100) };
        assert_eq!(cache_len, 6);
        assert_eq!(cache[4], 0x22);
        assert_eq!(cache[5], 0x22);
    }

    #[test]
    fn kv_cache_multi_append() {
        let mut cache = Vec::new();
        let mut cache_len = 0usize;
        for i in 0..5 {
            let data = vec![i as u8; 3];
            unsafe { neon_kv_cache_append_i2(&mut cache, &data, &mut cache_len, 100) };
        }
        assert_eq!(cache_len, 15);
        assert_eq!(cache[0], 0);
        assert_eq!(cache[3], 1);
        assert_eq!(cache[12], 4);
    }

    #[test]
    fn kv_cache_max_len_boundary() {
        let mut cache = Vec::new();
        let mut cache_len = 0usize;
        let data = vec![0xFF; 10];
        unsafe { neon_kv_cache_append_i2(&mut cache, &data, &mut cache_len, 10) };
        assert_eq!(cache_len, 10);
        // Second append should be fully clipped
        let data2 = vec![0xAA; 5];
        unsafe { neon_kv_cache_append_i2(&mut cache, &data2, &mut cache_len, 10) };
        assert_eq!(cache_len, 10); // unchanged
    }

    #[test]
    fn kv_cache_overflow_handling() {
        let mut cache = vec![0x11; 8];
        let mut cache_len = 8usize;
        let data = vec![0x22; 10];
        unsafe { neon_kv_cache_append_i2(&mut cache, &data, &mut cache_len, 12) };
        // Only 4 bytes fit
        assert_eq!(cache_len, 12);
        assert_eq!(cache[8], 0x22);
        assert_eq!(cache[11], 0x22);
    }

    #[test]
    fn kv_cache_size_tracking() {
        let mut cache = Vec::new();
        let mut cache_len = 0usize;
        unsafe { neon_kv_cache_append_i2(&mut cache, &[1, 2, 3], &mut cache_len, 100) };
        assert_eq!(cache_len, 3);
        unsafe { neon_kv_cache_append_i2(&mut cache, &[4, 5], &mut cache_len, 100) };
        assert_eq!(cache_len, 5);
        unsafe { neon_kv_cache_append_i2(&mut cache, &[], &mut cache_len, 100) };
        assert_eq!(cache_len, 5);
    }

    // ── Integration tests ──────────────────────────────────────────────

    #[test]
    fn full_forward_tiny() {
        // 1 head, seq_len=1, head_dim=4
        let queries = vec![1.0, 0.0, 0.0, 0.0];
        let keys = pack_slice(&[1, 0, 0, 0]);
        let values = pack_slice(&[0, 0, 0, 1]);
        let result =
            unsafe { neon_multi_head_attention_i2(&queries, &keys, &values, 1.0, 1.0, 1, 1, 4) };
        assert_eq!(result.len(), 4);
        // Single position: softmax([dot]) = [1.0], output = v0 * 1.0
        assert!((result[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn full_forward_small() {
        // 2 heads, seq_len=2, head_dim=4
        let num_heads = 2;
        let seq_len = 2;
        let head_dim = 4;
        let queries = vec![1.0; num_heads * seq_len * head_dim];
        let keys = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let values = pack_slice(&vec![1i8; num_heads * seq_len * head_dim]);
        let result = unsafe {
            neon_multi_head_attention_i2(
                &queries, &keys, &values, 1.0, 1.0, num_heads, seq_len, head_dim,
            )
        };
        assert_eq!(result.len(), num_heads * seq_len * head_dim);
        for v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn full_forward_deterministic_output() {
        let queries = vec![0.5, -0.5, 0.3, -0.3];
        let keys = pack_slice(&[1, -1, 0, 1]);
        let values = pack_slice(&[-1, 1, -1, 1]);
        let r1 =
            unsafe { neon_multi_head_attention_i2(&queries, &keys, &values, 1.0, 1.0, 1, 1, 4) };
        let r2 =
            unsafe { neon_multi_head_attention_i2(&queries, &keys, &values, 1.0, 1.0, 1, 1, 4) };
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((a - b).abs() < 1e-10, "non-deterministic: {a} vs {b}");
        }
    }

    #[test]
    fn full_forward_numerical_precision() {
        // Large scale factors shouldn't produce NaN/Inf in final output
        let queries = vec![100.0; 4];
        let keys = pack_slice(&[1, 1, 1, 1]);
        let values = pack_slice(&[1, -1, 1, -1]);
        let result =
            unsafe { neon_multi_head_attention_i2(&queries, &keys, &values, 0.01, 1.0, 1, 1, 4) };
        for v in &result {
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    // ── Edge case tests ────────────────────────────────────────────────

    #[test]
    fn edge_seq_len_1() {
        let result = unsafe {
            neon_qk_dot_i2_f32(&[1.0, 2.0, 3.0, 4.0], &pack_slice(&[1, 1, 1, 1]), 1.0, 1, 4)
        };
        assert_eq!(result.len(), 1);
        assert!((result[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn edge_head_dim_1() {
        let hd = 1;
        let sl = 2;
        let queries = vec![3.0, 5.0];
        // packed_dim = ceil(1/4) = 1 byte per key position
        let keys = vec![pack_i2s(1, 0, 0, 0), pack_i2s(-1, 0, 0, 0)];
        let result = unsafe { neon_qk_dot_i2_f32(&queries, &keys, 1.0, sl, hd) };
        // q0·k0 = 3*1 = 3, q0·k1 = 3*(-1) = -3
        // q1·k0 = 5*1 = 5, q1·k1 = 5*(-1) = -5
        assert!((result[0] - 3.0).abs() < 1e-5);
        assert!((result[1] - (-3.0)).abs() < 1e-5);
        assert!((result[2] - 5.0).abs() < 1e-5);
        assert!((result[3] - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn edge_single_head_single_token() {
        let result = unsafe {
            neon_multi_head_attention_i2(
                &[1.0, 0.0, -1.0, 0.0],
                &pack_slice(&[0, 1, 0, -1]),
                &pack_slice(&[1, 1, 1, 1]),
                1.0,
                1.0,
                1,
                1,
                4,
            )
        };
        // Single token: softmax of single score = 1.0
        // Output = 1.0 * [1,1,1,1] * 1.0 = [1,1,1,1]
        for v in &result {
            assert!((*v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    // ── Decode helper tests ────────────────────────────────────────────

    #[test]
    fn decode_i2s_values() {
        assert_eq!(decode_i2s(0b00), 0);
        assert_eq!(decode_i2s(0b01), 1);
        assert_eq!(decode_i2s(0b11), -1);
        assert_eq!(decode_i2s(0b10), 0);
    }

    #[test]
    fn decode_i2s_f32_values() {
        assert_eq!(decode_i2s_f32(0b00), 0.0);
        assert_eq!(decode_i2s_f32(0b01), 1.0);
        assert_eq!(decode_i2s_f32(0b11), -1.0);
        assert_eq!(decode_i2s_f32(0b10), 0.0);
    }

    #[test]
    fn unpack_byte_round_trip() {
        let original = [1i8, -1, 0, 1];
        let packed = pack_i2s(original[0], original[1], original[2], original[3]);
        let unpacked = unpack_byte_i2s(packed);
        assert_eq!(unpacked, original);
    }

    #[test]
    fn pack_slice_round_trip() {
        let vals: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, 0];
        let packed = pack_slice(&vals);
        assert_eq!(packed.len(), 2);
        for (i, &v) in vals.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_idx = (i % 4) * 2;
            let decoded = decode_i2s(packed[byte_idx] >> bit_idx);
            assert_eq!(decoded, v, "mismatch at index {i}");
        }
    }

    // ── Scalar reference agreement tests ───────────────────────────────

    #[test]
    fn qk_dot_matches_scalar_reference() {
        let hd = 8;
        let sl = 3;
        let q: Vec<f32> = (0..sl * hd).map(|i| (i as f32) * 0.1).collect();
        let k_vals: Vec<i8> = (0..sl * hd)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let k = pack_slice(&k_vals);
        let scale = 0.75;
        let expected = reference_qk_dot(&q, &k, scale, sl, hd);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, scale, sl, hd) };
        for (i, (&e, &r)) in expected.iter().zip(result.iter()).enumerate() {
            assert!((e - r).abs() < 1e-4, "pos {i}: expected {e}, got {r}");
        }
    }

    #[test]
    fn softmax_row_basic() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        let mut ref_row = row.clone();
        unsafe { neon_softmax_row(&mut row) };
        reference_softmax(&mut ref_row);
        for (i, (&a, &b)) in row.iter().zip(ref_row.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "softmax {i}: {a} vs {b}");
        }
    }

    #[test]
    fn softmax_row_with_neg_inf() {
        let mut row = vec![1.0, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
        unsafe { neon_softmax_row(&mut row) };
        assert!(row[1] < 1e-10);
        assert!(row[3] < 1e-10);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_row_all_neg_inf() {
        let mut row = vec![f32::NEG_INFINITY; 4];
        unsafe { neon_softmax_row(&mut row) };
        // NaN is acceptable for all-neg-inf input; just don't panic
    }

    #[test]
    fn softmax_row_single_element() {
        let mut row = vec![5.0];
        unsafe { neon_softmax_row(&mut row) };
        assert!((row[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn kv_cache_large_append() {
        let mut cache = Vec::new();
        let mut cache_len = 0usize;
        let data = vec![0xBB; 64];
        unsafe { neon_kv_cache_append_i2(&mut cache, &data, &mut cache_len, 1000) };
        assert_eq!(cache_len, 64);
        for i in 0..64 {
            assert_eq!(cache[i], 0xBB);
        }
    }

    #[test]
    fn qk_dot_non_aligned_head_dim() {
        // head_dim=5 → not a multiple of 4
        let hd = 5;
        let sl = 1;
        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k_vals: Vec<i8> = vec![1, -1, 1, -1, 1];
        let k = pack_slice(&k_vals);
        let result = unsafe { neon_qk_dot_i2_f32(&q, &k, 1.0, sl, hd) };
        // 1*1 + 2*(-1) + 3*1 + 4*(-1) + 5*1 = 1 - 2 + 3 - 4 + 5 = 3.0
        assert!((result[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_sum_non_aligned_head_dim() {
        let seq_len = 1;
        let head_dim = 5;
        let weights = vec![1.0];
        let values = pack_slice(&[1, -1, 1, -1, 1]);
        let result = unsafe { neon_weighted_sum_i2_f32(&weights, &values, 1.0, seq_len, head_dim) };
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - (-1.0)).abs() < 1e-5);
        assert!((result[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn attention_score_seq_len_1_no_mask() {
        let mut scores = vec![5.0];
        unsafe { neon_attention_score_f32(&mut scores, 1, 4) };
        let expected = 5.0 / 2.0;
        assert!((scores[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn multi_head_scale_propagation() {
        // Verify that key_scale and value_scale affect output
        let q = vec![1.0; 4];
        let k = pack_slice(&[1, 1, 1, 1]);
        let v = pack_slice(&[1, 1, 1, 1]);
        let r1 = unsafe { neon_multi_head_attention_i2(&q, &k, &v, 1.0, 1.0, 1, 1, 4) };
        let r2 = unsafe { neon_multi_head_attention_i2(&q, &k, &v, 1.0, 2.0, 1, 1, 4) };
        // value_scale=2 should double the output vs value_scale=1
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!((b - 2.0 * a).abs() < 1e-4, "{b} vs 2*{a}");
        }
    }
}
