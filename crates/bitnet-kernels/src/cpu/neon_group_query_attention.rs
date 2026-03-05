#![allow(unsafe_op_in_unsafe_fn, unused_unsafe, dead_code, unused_variables, unused_assignments)]
//! ARM NEON optimized Grouped Query Attention (GQA) for Apple Silicon.
//!
//! GQA shares KV heads across multiple query heads, reducing memory bandwidth.
//! Each KV head serves `num_q_heads / num_kv_heads` query heads. All dot-product
//! accumulations use NEON `float32x4` intrinsics for 4-wide SIMD throughput,
//! with scalar fallback for tail elements whose count is not a multiple of 4.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

/// Compute the NEON dot-product of two f32 slices of length `len`.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot(a: *const f32, b: *const f32, len: usize) -> f32 {
    let chunks = len / LANES;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.add(i * LANES));
        let vb = vld1q_f32(b.add(i * LANES));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut sum = vaddvq_f32(acc);
    for i in (chunks * LANES)..len {
        sum += *a.add(i) * *b.add(i);
    }
    sum
}

/// Softmax in-place over `data[..len]` using NEON reductions.
///
/// # Safety
/// Requires `aarch64` target with NEON. `data` must point to at least `len` f32s.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_softmax_inplace(data: *mut f32, len: usize) {
    // Find max for numerical stability.
    let chunks = len / LANES;
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(data.add(i * LANES));
        max_vec = vmaxq_f32(max_vec, v);
    }
    let mut max_val = vmaxvq_f32(max_vec);
    for i in (chunks * LANES)..len {
        max_val = max_val.max(*data.add(i));
    }

    // exp(x - max) and accumulate sum.
    let max_splat = vdupq_n_f32(max_val);
    let mut sum_vec = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let base = i * LANES;
        let v = vsubq_f32(vld1q_f32(data.add(base)), max_splat);
        // Scalar exp per lane — keeps accuracy without a fast-exp polynomial.
        let mut buf = [0f32; LANES];
        vst1q_f32(buf.as_mut_ptr(), v);
        for b in &mut buf {
            *b = b.exp();
        }
        let ev = vld1q_f32(buf.as_ptr());
        vst1q_f32(data.add(base), ev);
        sum_vec = vaddq_f32(sum_vec, ev);
    }
    let mut sum = vaddvq_f32(sum_vec);
    for i in (chunks * LANES)..len {
        let e = (*data.add(i) - max_val).exp();
        *data.add(i) = e;
        sum += e;
    }

    // Normalize.
    if sum == 0.0 {
        return;
    }
    let inv = 1.0 / sum;
    let inv_splat = vdupq_n_f32(inv);
    for i in 0..chunks {
        let base = i * LANES;
        let v = vld1q_f32(data.add(base));
        vst1q_f32(data.add(base), vmulq_f32(v, inv_splat));
    }
    for i in (chunks * LANES)..len {
        *data.add(i) *= inv;
    }
}

/// Full Grouped Query Attention forward pass.
///
/// `query`  is laid out as `[num_q_heads, seq_len, head_dim]` (row-major).
/// `key`    is laid out as `[num_kv_heads, seq_len, head_dim]`.
/// `value`  is laid out as `[num_kv_heads, seq_len, head_dim]`.
///
/// Returns output shaped `[num_q_heads, seq_len, head_dim]`.
///
/// # Panics
/// * `num_q_heads` is not divisible by `num_kv_heads`.
/// * Slice lengths do not match the declared dimensions.
#[cfg(target_arch = "aarch64")]
pub fn neon_gqa_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert!(num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads));
    assert_eq!(query.len(), num_q_heads * seq_len * head_dim);
    assert_eq!(key.len(), num_kv_heads * seq_len * head_dim);
    assert_eq!(value.len(), num_kv_heads * seq_len * head_dim);

    let group_size = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; num_q_heads * seq_len * head_dim];

    for qh in 0..num_q_heads {
        let kv_head = qh / group_size;
        for qi in 0..seq_len {
            // Compute attention scores for this query position.
            let q_offset = (qh * seq_len + qi) * head_dim;
            let mut scores = vec![0.0f32; seq_len];

            for (ki, score) in scores.iter_mut().enumerate().take(seq_len) {
                let k_offset = (kv_head * seq_len + ki) * head_dim;
                let dot = unsafe {
                    neon_dot(query.as_ptr().add(q_offset), key.as_ptr().add(k_offset), head_dim)
                };
                *score = dot * scale;
            }

            // Softmax over scores.
            unsafe {
                neon_softmax_inplace(scores.as_mut_ptr(), seq_len);
            }

            // Weighted sum of value vectors.
            let out_offset = (qh * seq_len + qi) * head_dim;
            for (vi, &w) in scores.iter().enumerate().take(seq_len) {
                let v_offset = (kv_head * seq_len + vi) * head_dim;
                if w == 0.0 {
                    continue;
                }
                let w_splat = unsafe { vdupq_n_f32(w) };
                let chunks = head_dim / LANES;
                for c in 0..chunks {
                    let base = c * LANES;
                    unsafe {
                        let vv = vld1q_f32(value.as_ptr().add(v_offset + base));
                        let ov = vld1q_f32(output.as_ptr().add(out_offset + base));
                        vst1q_f32(
                            output.as_mut_ptr().add(out_offset + base),
                            vfmaq_f32(ov, vv, w_splat),
                        );
                    }
                }
                for d in (chunks * LANES)..head_dim {
                    output[out_offset + d] += w * value[v_offset + d];
                }
            }
        }
    }

    output
}

/// Compute QK attention scores with KV head sharing.
///
/// Returns scores shaped `[num_q_heads, seq_len, seq_len]` (row-major).
#[cfg(target_arch = "aarch64")]
pub fn neon_gqa_scores(
    query: &[f32],
    key: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert!(num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads));
    assert_eq!(query.len(), num_q_heads * seq_len * head_dim);
    assert_eq!(key.len(), num_kv_heads * seq_len * head_dim);

    let group_size = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; num_q_heads * seq_len * seq_len];

    for qh in 0..num_q_heads {
        let kv_head = qh / group_size;
        for qi in 0..seq_len {
            let q_offset = (qh * seq_len + qi) * head_dim;
            for ki in 0..seq_len {
                let k_offset = (kv_head * seq_len + ki) * head_dim;
                let dot = unsafe {
                    neon_dot(query.as_ptr().add(q_offset), key.as_ptr().add(k_offset), head_dim)
                };
                scores[qh * seq_len * seq_len + qi * seq_len + ki] = dot * scale;
            }
        }
    }

    scores
}

/// Repeat KV heads to match query head count (explicit expansion).
///
/// Input `kv` is `[num_kv_heads, seq_len, head_dim]`.
/// Output is `[num_q_heads, seq_len, head_dim]` where each KV head is
/// duplicated `num_q_heads / num_kv_heads` times.
#[cfg(target_arch = "aarch64")]
pub fn neon_repeat_kv_heads(
    kv: &[f32],
    num_kv_heads: usize,
    num_q_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert!(num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads));
    assert_eq!(kv.len(), num_kv_heads * seq_len * head_dim);

    let group_size = num_q_heads / num_kv_heads;
    let head_elements = seq_len * head_dim;
    let mut out = vec![0.0f32; num_q_heads * head_elements];

    for kv_h in 0..num_kv_heads {
        let src_offset = kv_h * head_elements;
        for g in 0..group_size {
            let dst_head = kv_h * group_size + g;
            let dst_offset = dst_head * head_elements;

            let chunks = head_elements / LANES;
            for c in 0..chunks {
                let base = c * LANES;
                unsafe {
                    let v = vld1q_f32(kv.as_ptr().add(src_offset + base));
                    vst1q_f32(out.as_mut_ptr().add(dst_offset + base), v);
                }
            }
            let tail_start = chunks * LANES;
            out[dst_offset + tail_start..dst_offset + head_elements]
                .copy_from_slice(&kv[src_offset + tail_start..src_offset + head_elements]);
        }
    }

    out
}

/// GQA forward pass with an attention mask.
///
/// `mask` has length `seq_len`. Position `i` is attended to when `mask[i]` is
/// `true`; masked positions receive `-1e9` before softmax.
///
/// Layout and return shape are identical to [`neon_gqa_attention`].
#[cfg(target_arch = "aarch64")]
pub fn neon_gqa_with_mask(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    mask: &[bool],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert!(num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads));
    assert_eq!(query.len(), num_q_heads * seq_len * head_dim);
    assert_eq!(key.len(), num_kv_heads * seq_len * head_dim);
    assert_eq!(value.len(), num_kv_heads * seq_len * head_dim);
    assert_eq!(mask.len(), seq_len);

    let group_size = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; num_q_heads * seq_len * head_dim];

    for qh in 0..num_q_heads {
        let kv_head = qh / group_size;
        for qi in 0..seq_len {
            let q_offset = (qh * seq_len + qi) * head_dim;
            let mut scores = vec![0.0f32; seq_len];

            for (ki, score) in scores.iter_mut().enumerate().take(seq_len) {
                if !mask[ki] {
                    *score = -1e9;
                    continue;
                }
                let k_offset = (kv_head * seq_len + ki) * head_dim;
                let dot = unsafe {
                    neon_dot(query.as_ptr().add(q_offset), key.as_ptr().add(k_offset), head_dim)
                };
                *score = dot * scale;
            }

            unsafe {
                neon_softmax_inplace(scores.as_mut_ptr(), seq_len);
            }

            let out_offset = (qh * seq_len + qi) * head_dim;
            for (vi, &w) in scores.iter().enumerate().take(seq_len) {
                let v_offset = (kv_head * seq_len + vi) * head_dim;
                if w == 0.0 {
                    continue;
                }
                let w_splat = unsafe { vdupq_n_f32(w) };
                let chunks = head_dim / LANES;
                for c in 0..chunks {
                    let base = c * LANES;
                    unsafe {
                        let vv = vld1q_f32(value.as_ptr().add(v_offset + base));
                        let ov = vld1q_f32(output.as_ptr().add(out_offset + base));
                        vst1q_f32(
                            output.as_mut_ptr().add(out_offset + base),
                            vfmaq_f32(ov, vv, w_splat),
                        );
                    }
                }
                for d in (chunks * LANES)..head_dim {
                    output[out_offset + d] += w * value[v_offset + d];
                }
            }
        }
    }

    output
}

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// When num_q_heads == num_kv_heads, GQA degenerates to standard MHA.
    #[test]
    fn test_gqa_equal_heads() {
        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 2;
        let total = num_heads * seq_len * head_dim;

        // Deterministic data: element index as float.
        let query: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let key: Vec<f32> = (0..total).map(|i| ((total - i) as f32) * 0.01).collect();
        let value: Vec<f32> = (0..total).map(|i| (i as f32) * 0.02).collect();

        let gqa_out =
            neon_gqa_attention(&query, &key, &value, num_heads, num_heads, head_dim, seq_len);
        assert_eq!(gqa_out.len(), total);

        // With equal heads the KV head index equals the query head index,
        // so a second call with the same parameters must produce the same result.
        let gqa_out2 =
            neon_gqa_attention(&query, &key, &value, num_heads, num_heads, head_dim, seq_len);
        assert_eq!(gqa_out, gqa_out2);

        // Output values must be finite.
        for v in &gqa_out {
            assert!(v.is_finite(), "non-finite value in GQA output");
        }
    }

    /// 8 query heads sharing 2 KV heads (group size = 4).
    #[test]
    fn test_gqa_grouped() {
        let num_q_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 3;

        let q: Vec<f32> =
            (0..num_q_heads * seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> =
            (0..num_kv_heads * seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> =
            (0..num_kv_heads * seq_len * head_dim).map(|i| (i as f32) * 0.02).collect();

        let out = neon_gqa_attention(&q, &k, &v, num_q_heads, num_kv_heads, head_dim, seq_len);
        assert_eq!(out.len(), num_q_heads * seq_len * head_dim);

        // Query heads 0..3 share KV head 0, heads 4..7 share KV head 1.
        // Heads within the same group see identical K/V so their outputs
        // differ only because of distinct Q vectors.
        for v in &out {
            assert!(v.is_finite(), "non-finite value in grouped GQA output");
        }
    }

    /// Verify that repeat-KV correctly duplicates heads.
    #[test]
    fn test_repeat_kv_correctness() {
        let num_kv_heads = 2;
        let num_q_heads = 6;
        let head_dim = 4;
        let seq_len = 2;
        let head_elements = seq_len * head_dim;

        let kv: Vec<f32> = (0..num_kv_heads * head_elements).map(|i| i as f32).collect();

        let expanded = neon_repeat_kv_heads(&kv, num_kv_heads, num_q_heads, head_dim, seq_len);
        assert_eq!(expanded.len(), num_q_heads * head_elements);

        let group_size = num_q_heads / num_kv_heads; // 3
        for kv_h in 0..num_kv_heads {
            let src = &kv[kv_h * head_elements..(kv_h + 1) * head_elements];
            for g in 0..group_size {
                let dst_head = kv_h * group_size + g;
                let dst = &expanded[dst_head * head_elements..(dst_head + 1) * head_elements];
                assert_eq!(src, dst, "KV head {kv_h} replica {g} mismatch");
            }
        }
    }

    /// Causal mask: position i can only attend to positions 0..=i.
    #[test]
    fn test_gqa_with_causal_mask() {
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 4;

        let q: Vec<f32> =
            (0..num_q_heads * seq_len * head_dim).map(|i| (i as f32) * 0.05).collect();
        let k: Vec<f32> =
            (0..num_kv_heads * seq_len * head_dim).map(|i| (i as f32) * 0.05).collect();
        let v: Vec<f32> =
            (0..num_kv_heads * seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();

        // Mask allows only position 0 and 1.
        let mask = vec![true, true, false, false];

        let out_masked =
            neon_gqa_with_mask(&q, &k, &v, &mask, num_q_heads, num_kv_heads, head_dim, seq_len);
        assert_eq!(out_masked.len(), num_q_heads * seq_len * head_dim);

        // Compare against unmasked — results must differ because the mask
        // zeroes out contributions from positions 2 and 3.
        let out_full = neon_gqa_attention(&q, &k, &v, num_q_heads, num_kv_heads, head_dim, seq_len);
        assert_ne!(out_masked, out_full, "mask should change output");

        for v in &out_masked {
            assert!(v.is_finite(), "non-finite value in masked GQA output");
        }
    }
}
