//! ARM NEON optimized flash attention for Apple Silicon
//!
//! Memory-efficient attention using a tiled approach that processes K/V in
//! blocks, maintaining running statistics (max, sum) for numerically stable
//! softmax without materializing the full attention matrix.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Scalar fast exp (degree-4 polynomial, adequate for softmax normalisation).
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON vectorised fast exp.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);

    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c4 = vdupq_n_f32(1.0 / 24.0);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c2 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let poly = vfmaq_f32(c3, c4, r);
    let poly = vfmaq_f32(c2, poly, r);
    let poly = vfmaq_f32(one, poly, r);
    let poly = vfmaq_f32(one, poly, r);

    // 2^n via integer bit manipulation on each lane.
    let ni = vcvtq_s32_f32(n);
    let bias = vdupq_n_s32(127);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ni, bias)));

    vmulq_f32(poly, pow2n)
}

/// Horizontal sum of a `float32x4_t`.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

/// NEON-accelerated dot product of two slices (with scalar tail).
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / LANES;
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * LANES));
        let vb = vld1q_f32(b.as_ptr().add(i * LANES));
        acc = vfmaq_f32(acc, va, vb);
    }
    let mut sum = hsum_f32x4(acc);
    for i in (chunks * LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// NEON-accelerated fused multiply-add: `out[i] += scale * vec[i]`.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scaled_add(out: &mut [f32], vec: &[f32], scale: f32, len: usize) {
    let vs = vdupq_n_f32(scale);
    let chunks = len / LANES;
    for i in 0..chunks {
        let vo = vld1q_f32(out.as_ptr().add(i * LANES));
        let vv = vld1q_f32(vec.as_ptr().add(i * LANES));
        vst1q_f32(out.as_mut_ptr().add(i * LANES), vfmaq_f32(vo, vs, vv));
    }
    for i in (chunks * LANES)..len {
        out[i] += scale * vec[i];
    }
}

// ── Public API ──────────────────────────────────────────────────────────

/// Online softmax returning `(max, sum)` for flash attention correction.
///
/// Updates `scores[0..len]` in-place to `exp(scores[i] - max)` and returns
/// the running max and normalisation sum. Caller divides by `sum` to obtain
/// the final softmax distribution.
///
/// # Safety
/// Internally uses NEON intrinsics on `aarch64`.
#[cfg(target_arch = "aarch64")]
pub fn neon_attention_softmax_online(scores: &mut [f32], len: usize) -> (f32, f32) {
    assert!(len <= scores.len(), "len exceeds scores length");
    if len == 0 {
        return (f32::NEG_INFINITY, 0.0);
    }

    // Pass 1: find max.
    let max_val = scores[..len].iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Pass 2: exp(x - max) and accumulate sum via NEON.
    // Safety: guarded by `#[cfg(target_arch = "aarch64")]`.
    unsafe { online_softmax_inner(scores, len, max_val) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn online_softmax_inner(scores: &mut [f32], len: usize, max_val: f32) -> (f32, f32) {
    let vmax = vdupq_n_f32(max_val);
    let mut vsum = vdupq_n_f32(0.0);
    let chunks = len / LANES;

    for i in 0..chunks {
        let ptr = scores.as_mut_ptr().add(i * LANES);
        let v = vld1q_f32(ptr);
        let e = fast_exp_neon(vsubq_f32(v, vmax));
        vst1q_f32(ptr, e);
        vsum = vaddq_f32(vsum, e);
    }

    let mut sum = hsum_f32x4(vsum);
    for s in scores.iter_mut().take(len).skip(chunks * LANES) {
        let e = fast_exp_scalar(*s - max_val);
        *s = e;
        sum += e;
    }

    (max_val, sum)
}

/// Flash attention forward pass with tiled softmax (online softmax with running max).
///
/// Processes K/V in blocks of `block_size` rows, maintaining running max and
/// sum for numerically stable softmax without materialising the full `seq_len ×
/// seq_len` attention matrix.
///
/// # Layout
/// * `query`, `key`, `value`: row-major `[seq_len, head_dim]`
/// * Returns row-major `[seq_len, head_dim]`
///
/// # Safety
/// Internally uses NEON intrinsics on `aarch64`.
#[cfg(target_arch = "aarch64")]
pub fn neon_flash_attention_forward(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    head_dim: usize,
    seq_len: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), seq_len * head_dim);
    assert_eq!(key.len(), seq_len * head_dim);
    assert_eq!(value.len(), seq_len * head_dim);
    assert!(block_size > 0, "block_size must be > 0");

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for q_row in 0..seq_len {
        let q = &query[q_row * head_dim..(q_row + 1) * head_dim];
        let out = &mut output[q_row * head_dim..(q_row + 1) * head_dim];

        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f32;

        for blk_start in (0..seq_len).step_by(block_size) {
            let blk_end = (blk_start + block_size).min(seq_len);

            // Compute scores for this block and find block max.
            let blk_len = blk_end - blk_start;
            let mut scores = Vec::with_capacity(blk_len);
            for k_row in blk_start..blk_end {
                let k = &key[k_row * head_dim..(k_row + 1) * head_dim];
                // Safety: guarded by outer `#[cfg(target_arch = "aarch64")]`.
                let dot = unsafe { neon_dot(q, k, head_dim) };
                scores.push(dot * scale);
            }

            let blk_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Online correction: rescale previous accumulation.
            let prev_max = global_max;
            global_max = global_max.max(blk_max);
            if global_sum > 0.0 {
                let correction = fast_exp_scalar(prev_max - global_max);
                global_sum *= correction;
                for o in out.iter_mut() {
                    *o *= correction;
                }
            }

            // Accumulate softmax-weighted values for this block.
            let mut blk_sum = 0.0f32;
            for (idx, &s) in scores.iter().enumerate() {
                let w = fast_exp_scalar(s - global_max);
                blk_sum += w;
                let v_row = &value[(blk_start + idx) * head_dim..(blk_start + idx + 1) * head_dim];
                // Safety: guarded by outer `#[cfg(target_arch = "aarch64")]`.
                unsafe {
                    neon_scaled_add(out, v_row, w, head_dim);
                }
            }
            global_sum += blk_sum;
        }

        // Final normalisation.
        if global_sum > 0.0 {
            let inv = 1.0 / global_sum;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
    }

    output
}

/// Causal (autoregressive) flash attention.
///
/// Identical to [`neon_flash_attention_forward`] but applies a causal mask so
/// that position `i` can only attend to positions `0..=i`.
///
/// # Safety
/// Internally uses NEON intrinsics on `aarch64`.
#[cfg(target_arch = "aarch64")]
pub fn neon_causal_flash_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    head_dim: usize,
    seq_len: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), seq_len * head_dim);
    assert_eq!(key.len(), seq_len * head_dim);
    assert_eq!(value.len(), seq_len * head_dim);
    assert!(block_size > 0, "block_size must be > 0");

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for q_row in 0..seq_len {
        let q = &query[q_row * head_dim..(q_row + 1) * head_dim];
        let out = &mut output[q_row * head_dim..(q_row + 1) * head_dim];

        let causal_len = q_row + 1; // can attend to 0..=q_row
        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f32;

        for blk_start in (0..causal_len).step_by(block_size) {
            let blk_end = (blk_start + block_size).min(causal_len);
            let blk_len = blk_end - blk_start;

            let mut scores = Vec::with_capacity(blk_len);
            for k_row in blk_start..blk_end {
                let k = &key[k_row * head_dim..(k_row + 1) * head_dim];
                let dot = unsafe { neon_dot(q, k, head_dim) };
                scores.push(dot * scale);
            }

            let blk_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let prev_max = global_max;
            global_max = global_max.max(blk_max);
            if global_sum > 0.0 {
                let correction = fast_exp_scalar(prev_max - global_max);
                global_sum *= correction;
                for o in out.iter_mut() {
                    *o *= correction;
                }
            }

            let mut blk_sum = 0.0f32;
            for (idx, &s) in scores.iter().enumerate() {
                let w = fast_exp_scalar(s - global_max);
                blk_sum += w;
                let v_row = &value[(blk_start + idx) * head_dim..(blk_start + idx + 1) * head_dim];
                unsafe {
                    neon_scaled_add(out, v_row, w, head_dim);
                }
            }
            global_sum += blk_sum;
        }

        if global_sum > 0.0 {
            let inv = 1.0 / global_sum;
            for o in out.iter_mut() {
                *o *= inv;
            }
        }
    }

    output
}

/// Multi-head attention wrapper using flash attention internally.
///
/// # Layout
/// * `query`, `key`, `value`: row-major `[num_heads, seq_len, head_dim]`
///   (heads are the outermost dimension).
/// * Returns row-major `[num_heads, seq_len, head_dim]`.
///
/// # Safety
/// Internally uses NEON intrinsics on `aarch64`.
#[cfg(target_arch = "aarch64")]
pub fn neon_multi_head_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    let head_elems = seq_len * head_dim;
    assert_eq!(query.len(), num_heads * head_elems);
    assert_eq!(key.len(), num_heads * head_elems);
    assert_eq!(value.len(), num_heads * head_elems);

    let block_size = 64.min(seq_len);
    let mut output = Vec::with_capacity(num_heads * head_elems);

    for h in 0..num_heads {
        let offset = h * head_elems;
        let q = &query[offset..offset + head_elems];
        let k = &key[offset..offset + head_elems];
        let v = &value[offset..offset + head_elems];
        let head_out = neon_flash_attention_forward(q, k, v, head_dim, seq_len, block_size);
        output.extend_from_slice(&head_out);
    }

    output
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    /// Naive (materialised) attention for reference: softmax(Q·Kᵀ / √d) · V.
    fn naive_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            let qi = &q[i * head_dim..(i + 1) * head_dim];
            // Compute scores.
            let mut scores: Vec<f32> = (0..seq_len)
                .map(|j| {
                    let kj = &k[j * head_dim..(j + 1) * head_dim];
                    qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            // Standard softmax.
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (s, e) in scores.iter_mut().zip(&exps) {
                *s = e / sum;
            }
            // Weighted sum.
            let out = &mut output[i * head_dim..(i + 1) * head_dim];
            for (j, &w) in scores.iter().enumerate() {
                let vj = &v[j * head_dim..(j + 1) * head_dim];
                for (o, &vv) in out.iter_mut().zip(vj) {
                    *o += w * vv;
                }
            }
        }
        output
    }

    #[test]
    fn test_flash_vs_naive_small() {
        let seq_len = 4;
        let head_dim = 8;
        // Deterministic inputs.
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1 - 1.6).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.05 - 0.8).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.02 + 0.1).collect();

        let naive = naive_attention(&q, &k, &v, head_dim, seq_len);
        let flash = neon_flash_attention_forward(&q, &k, &v, head_dim, seq_len, 2);

        for (i, (n, f)) in naive.iter().zip(&flash).enumerate() {
            assert!((n - f).abs() < 1e-3, "mismatch at index {i}: naive={n}, flash={f}");
        }
    }

    #[test]
    fn test_causal_mask() {
        let seq_len = 4;
        let head_dim = 4;
        let q = vec![1.0f32; seq_len * head_dim];
        let k = vec![1.0f32; seq_len * head_dim];
        // Give each V row a distinct value so we can detect which rows are attended.
        let mut v = vec![0.0f32; seq_len * head_dim];
        for row in 0..seq_len {
            for d in 0..head_dim {
                v[row * head_dim + d] = (row + 1) as f32;
            }
        }

        let out = neon_causal_flash_attention(&q, &k, &v, head_dim, seq_len, 2);

        // Row 0 attends only to V[0] → output ≈ 1.0
        for d in 0..head_dim {
            assert!(
                (out[d] - 1.0).abs() < 1e-3,
                "row 0 should attend only to V[0], got {}",
                out[d]
            );
        }
        // Row 3 attends to V[0..=3] with uniform Q/K → mean of 1,2,3,4 = 2.5
        for d in 0..head_dim {
            let val = out[3 * head_dim + d];
            assert!((val - 2.5).abs() < 1e-2, "row 3 should be ~2.5 (mean of 1..4), got {val}");
        }
    }

    #[test]
    fn test_multi_head_shape() {
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 8;
        let total = num_heads * seq_len * head_dim;
        let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 + 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 - 0.3).collect();

        let out = neon_multi_head_attention(&q, &k, &v, num_heads, head_dim, seq_len);
        assert_eq!(out.len(), total, "output length should match input layout");
        assert!(out.iter().all(|x| x.is_finite()), "all outputs should be finite");
    }

    #[test]
    fn test_online_softmax_vs_standard() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 3.5, 2.5];
        let len = scores.len();

        // Reference: standard softmax.
        let max_ref = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max_ref).exp()).collect();
        let sum_ref: f32 = exps.iter().sum();
        let reference: Vec<f32> = exps.iter().map(|e| e / sum_ref).collect();

        let (max_val, sum_val) = neon_attention_softmax_online(&mut scores, len);

        assert!((max_val - max_ref).abs() < 1e-6, "max mismatch: online={max_val}, ref={max_ref}");
        assert!((sum_val - sum_ref).abs() < 1e-2, "sum mismatch: online={sum_val}, ref={sum_ref}");

        // After division by sum, values should match standard softmax.
        for (i, (&s, &r)) in scores.iter().zip(&reference).enumerate() {
            let normalised = s / sum_val;
            assert!(
                (normalised - r).abs() < 1e-3,
                "softmax mismatch at {i}: online={normalised}, ref={r}"
            );
        }
    }
}
