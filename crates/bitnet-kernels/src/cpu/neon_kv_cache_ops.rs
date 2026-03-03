//! NEON-optimized KV cache operations for Apple Silicon autoregressive inference.

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
    clippy::excessive_precision
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const NEON_LANES: usize = 4;

// ── Scalar reference implementations ────────────────────────────────

/// Scalar: append `new_data` into `cache` at `write_pos` (in units of head_dim).
pub fn scalar_kv_cache_append(
    cache: &mut [f32],
    new_data: &[f32],
    write_pos: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let new_tokens = new_data.len() / (head_dim * num_heads);
    for h in 0..num_heads {
        for t in 0..new_tokens {
            let src_off = (t * num_heads + h) * head_dim;
            let dst_off =
                (h * cache_stride(cache, num_heads) + (write_pos + t) * head_dim) as usize;
            for d in 0..head_dim {
                cache[dst_off + d] = new_data[src_off + d];
            }
        }
    }
}

/// Scalar: gather cached K/V at specific `positions` into `output`.
pub fn scalar_kv_cache_gather(
    cache: &[f32],
    positions: &[u32],
    output: &mut [f32],
    head_dim: usize,
    num_heads: usize,
) {
    for h in 0..num_heads {
        for (pi, &pos) in positions.iter().enumerate() {
            let cache_off = h * cache_head_stride(cache, num_heads) + (pos as usize) * head_dim;
            let out_off = (pi * num_heads + h) * head_dim;
            for d in 0..head_dim {
                output[out_off + d] = cache[cache_off + d];
            }
        }
    }
}

/// Scalar: rotate cache as circular buffer by `shift` positions.
pub fn scalar_kv_cache_rotate(
    cache: &mut [f32],
    max_seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    shift: usize,
) {
    let mut tmp = vec![0.0f32; max_seq_len * head_dim];
    for h in 0..num_heads {
        let base = h * max_seq_len * head_dim;
        for t in 0..max_seq_len {
            let src_t = (t + shift) % max_seq_len;
            let src_off = base + src_t * head_dim;
            let tmp_off = t * head_dim;
            tmp[tmp_off..tmp_off + head_dim].copy_from_slice(&cache[src_off..src_off + head_dim]);
        }
        for t in 0..max_seq_len {
            let dst_off = base + t * head_dim;
            let tmp_off = t * head_dim;
            cache[dst_off..dst_off + head_dim].copy_from_slice(&tmp[tmp_off..tmp_off + head_dim]);
        }
    }
}

/// Scalar: concatenate existing cached K/V with new K/V along seq dim.
pub fn scalar_kv_cache_concat(
    existing: &[f32],
    new_data: &[f32],
    output: &mut [f32],
    existing_len: usize,
    new_len: usize,
    head_dim: usize,
    num_heads: usize,
) {
    for h in 0..num_heads {
        let ex_base = h * existing_len * head_dim;
        let out_base = h * (existing_len + new_len) * head_dim;
        for i in 0..existing_len * head_dim {
            output[out_base + i] = existing[ex_base + i];
        }
        let new_base = h * new_len * head_dim;
        let out_new_base = out_base + existing_len * head_dim;
        for i in 0..new_len * head_dim {
            output[out_new_base + i] = new_data[new_base + i];
        }
    }
}

/// Scalar: quantize f32 to i8 with per-head absmax scaling and store.
pub fn scalar_kv_cache_quantize_store(
    src: &[f32],
    dst_quant: &mut [i8],
    dst_scales: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) {
    for h in 0..num_heads {
        let base = h * seq_len * head_dim;
        for t in 0..seq_len {
            let off = base + t * head_dim;
            let mut amax: f32 = 0.0;
            for d in 0..head_dim {
                let v = src[off + d].abs();
                if v > amax {
                    amax = v;
                }
            }
            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            let inv_scale = 1.0 / scale;
            dst_scales[h * seq_len + t] = scale;
            for d in 0..head_dim {
                let q = (src[off + d] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                dst_quant[off + d] = q;
            }
        }
    }
}

/// Scalar: dequantize i8 K/V cache back to f32.
pub fn scalar_kv_cache_dequantize_load(
    src_quant: &[i8],
    src_scales: &[f32],
    dst: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) {
    for h in 0..num_heads {
        let base = h * seq_len * head_dim;
        for t in 0..seq_len {
            let off = base + t * head_dim;
            let scale = src_scales[h * seq_len + t];
            for d in 0..head_dim {
                dst[off + d] = (src_quant[off + d] as f32) * scale;
            }
        }
    }
}

// ── Layout helpers ──────────────────────────────────────────────────

#[inline(always)]
fn cache_stride(cache: &[f32], num_heads: usize) -> usize {
    cache.len() / num_heads
}

#[inline(always)]
fn cache_head_stride(cache: &[f32], num_heads: usize) -> usize {
    cache.len() / num_heads
}

// ── NEON-optimized implementations ──────────────────────────────────

/// Copy `len` f32 elements with NEON 4-wide loads/stores.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_copy_f32(src: *const f32, dst: *mut f32, len: usize) {
    let chunks = len / NEON_LANES;
    let rem = len % NEON_LANES;
    for i in 0..chunks {
        let off = i * NEON_LANES;
        let v = vld1q_f32(src.add(off));
        vst1q_f32(dst.add(off), v);
    }
    let tail = chunks * NEON_LANES;
    for i in 0..rem {
        *dst.add(tail + i) = *src.add(tail + i);
    }
}

/// Append new K/V vectors to the cache using NEON memcpy.
///
/// Layout: cache is `[num_heads, max_seq_len, head_dim]`.
/// `new_data` is `[new_tokens, num_heads, head_dim]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_append(
    cache: &mut [f32],
    new_data: &[f32],
    write_pos: usize,
    head_dim: usize,
    num_heads: usize,
    max_seq_len: usize,
) {
    let new_tokens = new_data.len() / (head_dim * num_heads);
    debug_assert!(write_pos + new_tokens <= max_seq_len);
    let head_stride = max_seq_len * head_dim;

    for h in 0..num_heads {
        for t in 0..new_tokens {
            let src_off = (t * num_heads + h) * head_dim;
            let dst_off = h * head_stride + (write_pos + t) * head_dim;
            neon_copy_f32(
                new_data.as_ptr().add(src_off),
                cache.as_mut_ptr().add(dst_off),
                head_dim,
            );
        }
    }
}

/// Gather cached K/V at specific positions using NEON loads.
///
/// Layout: cache `[num_heads, max_seq_len, head_dim]`,
///         output `[num_positions, num_heads, head_dim]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_gather(
    cache: &[f32],
    positions: &[u32],
    output: &mut [f32],
    head_dim: usize,
    num_heads: usize,
    max_seq_len: usize,
) {
    let head_stride = max_seq_len * head_dim;
    for h in 0..num_heads {
        for (pi, &pos) in positions.iter().enumerate() {
            let cache_off = h * head_stride + (pos as usize) * head_dim;
            let out_off = (pi * num_heads + h) * head_dim;
            neon_copy_f32(
                cache.as_ptr().add(cache_off),
                output.as_mut_ptr().add(out_off),
                head_dim,
            );
        }
    }
}

/// Rotate cache as circular buffer by `shift` positions using NEON.
///
/// After rotation, position `i` in the cache holds what was at `(i + shift) % max_seq_len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_rotate(
    cache: &mut [f32],
    max_seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    shift: usize,
) {
    let head_stride = max_seq_len * head_dim;
    let mut tmp = vec![0.0f32; max_seq_len * head_dim];

    for h in 0..num_heads {
        let base = h * head_stride;
        for t in 0..max_seq_len {
            let src_t = (t + shift) % max_seq_len;
            let src_off = base + src_t * head_dim;
            let tmp_off = t * head_dim;
            neon_copy_f32(cache.as_ptr().add(src_off), tmp.as_mut_ptr().add(tmp_off), head_dim);
        }
        neon_copy_f32(tmp.as_ptr(), cache.as_mut_ptr().add(base), max_seq_len * head_dim);
    }
}

/// Concatenate existing cached K/V with new K/V along the sequence dimension.
///
/// `existing` layout: `[num_heads, existing_len, head_dim]`
/// `new_data` layout: `[num_heads, new_len, head_dim]`
/// `output` layout: `[num_heads, existing_len + new_len, head_dim]`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_concat(
    existing: &[f32],
    new_data: &[f32],
    output: &mut [f32],
    existing_len: usize,
    new_len: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let out_seq = existing_len + new_len;
    for h in 0..num_heads {
        let ex_base = h * existing_len * head_dim;
        let out_base = h * out_seq * head_dim;
        neon_copy_f32(
            existing.as_ptr().add(ex_base),
            output.as_mut_ptr().add(out_base),
            existing_len * head_dim,
        );
        let new_base = h * new_len * head_dim;
        let out_new = out_base + existing_len * head_dim;
        neon_copy_f32(
            new_data.as_ptr().add(new_base),
            output.as_mut_ptr().add(out_new),
            new_len * head_dim,
        );
    }
}

/// Quantize f32 K/V to i8 with per-token absmax scaling, using NEON.
///
/// `src` layout: `[num_heads, seq_len, head_dim]`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_quantize_store(
    src: &[f32],
    dst_quant: &mut [i8],
    dst_scales: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let abs_mask = vdupq_n_u32(0x7FFF_FFFF);
    let clamp_lo = vdupq_n_f32(-128.0);
    let clamp_hi = vdupq_n_f32(127.0);

    for h in 0..num_heads {
        let base = h * seq_len * head_dim;
        for t in 0..seq_len {
            let off = base + t * head_dim;

            // Phase 1: find absmax via NEON
            let mut vmax = vdupq_n_f32(0.0);
            let chunks = head_dim / NEON_LANES;
            let rem = head_dim % NEON_LANES;
            for c in 0..chunks {
                let v = vld1q_f32(src.as_ptr().add(off + c * NEON_LANES));
                let a = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v), abs_mask));
                vmax = vmaxq_f32(vmax, a);
            }
            let mut amax = vmaxvq_f32(vmax);
            for r in 0..rem {
                let v = src[off + chunks * NEON_LANES + r].abs();
                if v > amax {
                    amax = v;
                }
            }

            let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
            let inv_scale = 1.0 / scale;
            dst_scales[h * seq_len + t] = scale;

            // Phase 2: quantize with NEON
            let vinv = vdupq_n_f32(inv_scale);
            for c in 0..chunks {
                let idx = off + c * NEON_LANES;
                let v = vld1q_f32(src.as_ptr().add(idx));
                let scaled = vmulq_f32(v, vinv);
                let rounded = vrndnq_f32(scaled);
                let clamped = vmaxq_f32(vminq_f32(rounded, clamp_hi), clamp_lo);
                // Extract and store as i8
                for lane in 0..NEON_LANES {
                    dst_quant[idx + lane] =
                        vgetq_lane_f32::<0>(vextq_f32(clamped, clamped, lane as u32)) as i8;
                }
            }
            // Scalar tail
            for r in 0..rem {
                let idx = off + chunks * NEON_LANES + r;
                let q = (src[idx] * inv_scale).round().clamp(-128.0, 127.0) as i8;
                dst_quant[idx] = q;
            }
        }
    }
}

/// Dequantize i8 K/V cache back to f32, using NEON.
///
/// `src_quant` layout: `[num_heads, seq_len, head_dim]`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_kv_cache_dequantize_load(
    src_quant: &[i8],
    src_scales: &[f32],
    dst: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
) {
    for h in 0..num_heads {
        let base = h * seq_len * head_dim;
        for t in 0..seq_len {
            let off = base + t * head_dim;
            let scale = src_scales[h * seq_len + t];
            let vscale = vdupq_n_f32(scale);

            let chunks = head_dim / NEON_LANES;
            let rem = head_dim % NEON_LANES;

            for c in 0..chunks {
                let idx = off + c * NEON_LANES;
                // Load 4 i8 values and widen to f32
                let mut vals = [0.0f32; NEON_LANES];
                for lane in 0..NEON_LANES {
                    vals[lane] = src_quant[idx + lane] as f32;
                }
                let vi = vld1q_f32(vals.as_ptr());
                let result = vmulq_f32(vi, vscale);
                vst1q_f32(dst.as_mut_ptr().add(idx), result);
            }
            for r in 0..rem {
                let idx = off + chunks * NEON_LANES + r;
                dst[idx] = (src_quant[idx] as f32) * scale;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;
    const QUANT_TOL: f32 = 0.15; // quantization round-trip tolerance

    fn assert_approx(a: f32, b: f32, tol: f32, ctx: &str) {
        let diff = (a - b).abs();
        assert!(diff <= tol, "{ctx}: {a} vs {b}, diff={diff}, tol={tol}");
    }

    fn assert_slices_approx(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch {} vs {}", a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_approx(x, y, tol, &format!("{ctx}[{i}]"));
        }
    }

    /// Make a cache buffer: [num_heads, max_seq_len, head_dim]
    fn make_cache(num_heads: usize, max_seq_len: usize, head_dim: usize) -> Vec<f32> {
        vec![0.0; num_heads * max_seq_len * head_dim]
    }

    /// Fill with sequential pattern for easy verification.
    fn fill_sequential(buf: &mut [f32], start: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = start + i as f32 * 0.01;
        }
    }

    /// Make new_data: [new_tokens, num_heads, head_dim]
    fn make_new_data(new_tokens: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
        let len = new_tokens * num_heads * head_dim;
        let mut data = vec![0.0; len];
        fill_sequential(&mut data, 1.0);
        data
    }

    // ================================================================
    // Append correctness (15+ tests)
    // ================================================================

    #[test]
    fn test_append_single_token_single_head() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = vec![1.0, 2.0, 3.0, 4.0]; // 1 token, 1 head, dim=4
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        assert_eq!(&cache[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cache[4..8], &[0.0; 4]); // rest untouched
    }

    #[test]
    fn test_append_single_token_multi_head() {
        let (heads, max_seq, hd) = (2, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        // [1 token, 2 heads, 4 dim] = 8 values: head0=[1,2,3,4], head1=[5,6,7,8]
        let new_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        // head0 starts at 0, head1 starts at 8*4=32
        assert_eq!(&cache[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cache[32..36], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        // 2 tokens, 1 head, 4 dim
        let new_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        assert_eq!(&cache[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cache[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_append_at_nonzero_write_pos() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = vec![10.0, 20.0, 30.0, 40.0];
        scalar_kv_cache_append(&mut cache, &new_data, 3, hd, heads);
        assert_eq!(&cache[0..4], &[0.0; 4]); // pos 0 untouched
        assert_eq!(&cache[12..16], &[10.0, 20.0, 30.0, 40.0]); // pos 3
    }

    #[test]
    fn test_append_fills_last_position() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = vec![9.0, 8.0, 7.0, 6.0];
        scalar_kv_cache_append(&mut cache, &new_data, 3, hd, heads);
        assert_eq!(&cache[12..16], &[9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn test_append_sequential_writes() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        for pos in 0..4 {
            let val = (pos + 1) as f32;
            let new_data = vec![val; hd];
            scalar_kv_cache_append(&mut cache, &new_data, pos, hd, heads);
        }
        for pos in 0..4 {
            let expected = (pos + 1) as f32;
            for d in 0..hd {
                assert_eq!(cache[pos * hd + d], expected);
            }
        }
    }

    #[test]
    fn test_append_large_head_dim() {
        let (heads, max_seq, hd) = (1, 4, 128);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data: Vec<f32> = (0..128).map(|i| i as f32).collect();
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        for d in 0..128 {
            assert_eq!(cache[d], d as f32);
        }
    }

    #[test]
    fn test_append_4_heads_dim64() {
        let (heads, max_seq, hd) = (4, 16, 64);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = make_new_data(1, heads, hd);
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        // Verify first element of each head in cache is correct
        let head_stride = max_seq * hd;
        for h in 0..heads {
            let src_off = h * hd;
            let dst_off = h * head_stride;
            assert_approx(cache[dst_off], new_data[src_off], TOL, "append 4head");
        }
    }

    #[test]
    fn test_append_does_not_clobber_other_positions() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let saved_pos1: Vec<f32> = cache[4..8].to_vec();
        let new_data = vec![99.0; hd];
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        assert_eq!(&cache[4..8], saved_pos1.as_slice());
    }

    #[test]
    fn test_append_two_tokens_two_heads() {
        let (heads, max_seq, hd) = (2, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        // [2 tokens, 2 heads, 4 dim] = 16 values
        // t0h0=[1..4], t0h1=[5..8], t1h0=[9..12], t1h1=[13..16]
        let new_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        let hs = max_seq * hd; // head stride
        assert_eq!(&cache[0..4], &[1.0, 2.0, 3.0, 4.0]); // h0, t0
        assert_eq!(&cache[4..8], &[9.0, 10.0, 11.0, 12.0]); // h0, t1
        assert_eq!(&cache[hs..hs + 4], &[5.0, 6.0, 7.0, 8.0]); // h1, t0
        assert_eq!(&cache[hs + 4..hs + 8], &[13.0, 14.0, 15.0, 16.0]); // h1, t1
    }

    #[test]
    fn test_append_overwrite_same_position() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let data1 = vec![1.0; hd];
        scalar_kv_cache_append(&mut cache, &data1, 0, hd, heads);
        let data2 = vec![2.0; hd];
        scalar_kv_cache_append(&mut cache, &data2, 0, hd, heads);
        assert_eq!(&cache[0..4], &[2.0; 4]);
    }

    #[test]
    fn test_append_negative_values() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let data = vec![-1.0, -2.5, -0.001, -100.0];
        scalar_kv_cache_append(&mut cache, &data, 0, hd, heads);
        assert_eq!(&cache[0..4], &[-1.0, -2.5, -0.001, -100.0]);
    }

    #[test]
    fn test_append_very_small_values() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let tiny = f32::MIN_POSITIVE;
        let data = vec![tiny; hd];
        scalar_kv_cache_append(&mut cache, &data, 0, hd, heads);
        assert_eq!(cache[0], tiny);
    }

    #[test]
    fn test_append_head_dim_32() {
        let (heads, max_seq, hd) = (2, 8, 32);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = make_new_data(1, heads, hd);
        scalar_kv_cache_append(&mut cache, &new_data, 5, hd, heads);
        let hs = max_seq * hd;
        // head 0, pos 5
        for d in 0..hd {
            assert_approx(cache[5 * hd + d], new_data[d], TOL, "h0 dim32");
        }
        // head 1, pos 5
        for d in 0..hd {
            assert_approx(cache[hs + 5 * hd + d], new_data[hd + d], TOL, "h1 dim32");
        }
    }

    // ================================================================
    // Gather/scatter operations (15+ tests)
    // ================================================================

    #[test]
    fn test_gather_single_position() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [2u32];
        let mut output = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &cache[8..12]);
    }

    #[test]
    fn test_gather_multiple_positions() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [0u32, 3, 7];
        let mut output = vec![0.0f32; 3 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &cache[0..4]);
        assert_eq!(&output[4..8], &cache[12..16]);
        assert_eq!(&output[8..12], &cache[28..32]);
    }

    #[test]
    fn test_gather_multi_head() {
        let (heads, max_seq, hd) = (2, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [1u32];
        let mut output = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        let hs = max_seq * hd;
        // output[0..4] = head0 pos1, output[4..8] = head1 pos1
        assert_eq!(&output[0..4], &cache[4..8]);
        assert_eq!(&output[4..8], &cache[hs + 4..hs + 8]);
    }

    #[test]
    fn test_gather_all_positions() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 1.0);
        let positions: Vec<u32> = (0..4).collect();
        let mut output = vec![0.0f32; 4 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_slices_approx(&output, &cache, TOL, "gather all");
    }

    #[test]
    fn test_gather_reverse_order() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [3u32, 2, 1, 0];
        let mut output = vec![0.0f32; 4 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &cache[12..16]);
        assert_eq!(&output[12..16], &cache[0..4]);
    }

    #[test]
    fn test_gather_duplicate_positions() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [2u32, 2, 2];
        let mut output = vec![0.0f32; 3 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &output[4..8]);
        assert_eq!(&output[4..8], &output[8..12]);
    }

    #[test]
    fn test_gather_last_position() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [7u32];
        let mut output = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &cache[28..32]);
    }

    #[test]
    fn test_gather_first_position() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 5.0);
        let positions = [0u32];
        let mut output = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &cache[0..4]);
    }

    #[test]
    fn test_gather_dim64() {
        let (heads, max_seq, hd) = (1, 4, 64);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [2u32];
        let mut output = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        let off = 2 * hd;
        assert_slices_approx(&output, &cache[off..off + hd], TOL, "gather dim64");
    }

    #[test]
    fn test_gather_dim128() {
        let (heads, max_seq, hd) = (1, 4, 128);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [1u32];
        let mut output = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        let off = 1 * hd;
        assert_slices_approx(&output, &cache[off..off + hd], TOL, "gather dim128");
    }

    #[test]
    fn test_gather_4_heads_2_positions() {
        let (heads, max_seq, hd) = (4, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [0u32, 5];
        let mut output = vec![0.0f32; 2 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        let hs = max_seq * hd;
        // First position (0), head0
        assert_eq!(&output[0..4], &cache[0..4]);
        // First position (0), head1
        assert_eq!(&output[4..8], &cache[hs..hs + 4]);
    }

    #[test]
    fn test_gather_preserves_cache() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        let positions = [1u32, 3, 5];
        let mut output = vec![0.0f32; 3 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(cache, orig);
    }

    #[test]
    fn test_gather_non_contiguous() {
        let (heads, max_seq, hd) = (1, 16, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [0u32, 5, 10, 15];
        let mut output = vec![0.0f32; 4 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        for (pi, &pos) in positions.iter().enumerate() {
            let co = (pos as usize) * hd;
            let oo = pi * hd;
            assert_eq!(&output[oo..oo + hd], &cache[co..co + hd]);
        }
    }

    #[test]
    fn test_gather_single_element_dim() {
        let (heads, max_seq, hd) = (1, 4, 1);
        let cache = vec![10.0, 20.0, 30.0, 40.0];
        let positions = [2u32];
        let mut output = vec![0.0f32; 1];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(output[0], 30.0);
    }

    #[test]
    fn test_gather_adjacent_positions() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let positions = [3u32, 4, 5];
        let mut output = vec![0.0f32; 3 * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        assert_eq!(&output[0..4], &cache[12..16]);
        assert_eq!(&output[4..8], &cache[16..20]);
        assert_eq!(&output[8..12], &cache[20..24]);
    }

    // ================================================================
    // Rotate/circular buffer (15+ tests)
    // ================================================================

    #[test]
    fn test_rotate_shift_0_is_identity() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 0);
        assert_eq!(cache, orig);
    }

    #[test]
    fn test_rotate_shift_1() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        // pos0=[1,1,1,1], pos1=[2,2,2,2], pos2=[3,3,3,3], pos3=[4,4,4,4]
        for t in 0..4 {
            for d in 0..4 {
                cache[t * 4 + d] = (t + 1) as f32;
            }
        }
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        // After shift=1: pos0 ← old pos1, pos1 ← old pos2, ...
        for d in 0..4 {
            assert_eq!(cache[0 * 4 + d], 2.0); // was pos1
            assert_eq!(cache[1 * 4 + d], 3.0); // was pos2
            assert_eq!(cache[2 * 4 + d], 4.0); // was pos3
            assert_eq!(cache[3 * 4 + d], 1.0); // was pos0 (wrapped)
        }
    }

    #[test]
    fn test_rotate_full_cycle() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, max_seq);
        assert_eq!(cache, orig); // full cycle = identity
    }

    #[test]
    fn test_rotate_shift_half() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = vec![0.0f32; 16];
        for t in 0..4 {
            for d in 0..4 {
                cache[t * 4 + d] = (t + 1) as f32;
            }
        }
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 2);
        // pos0 ← old pos2, pos1 ← old pos3, pos2 ← old pos0, pos3 ← old pos1
        assert_eq!(cache[0], 3.0);
        assert_eq!(cache[4], 4.0);
        assert_eq!(cache[8], 1.0);
        assert_eq!(cache[12], 2.0);
    }

    #[test]
    fn test_rotate_multi_head() {
        let (heads, max_seq, hd) = (2, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let hs = max_seq * hd;
        for t in 0..4 {
            for d in 0..4 {
                cache[t * 4 + d] = (t + 1) as f32;
                cache[hs + t * 4 + d] = (t + 1) as f32 + 10.0;
            }
        }
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        // head0: pos0 ← old pos1(=2)
        assert_eq!(cache[0], 2.0);
        // head1: pos0 ← old pos1(=12)
        assert_eq!(cache[hs], 12.0);
    }

    #[test]
    fn test_rotate_shift_max_minus_1() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = vec![0.0f32; 16];
        for t in 0..4 {
            for d in 0..4 {
                cache[t * 4 + d] = (t + 1) as f32;
            }
        }
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 3);
        // pos0 ← old pos3(=4), pos1 ← old pos0(=1)
        assert_eq!(cache[0], 4.0);
        assert_eq!(cache[4], 1.0);
    }

    #[test]
    fn test_rotate_double_rotate() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, max_seq - 1);
        assert_slices_approx(&cache, &orig, TOL, "double rotate");
    }

    #[test]
    fn test_rotate_dim64() {
        let (heads, max_seq, hd) = (1, 4, 64);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let pos1_orig: Vec<f32> = cache[hd..2 * hd].to_vec();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        assert_slices_approx(&cache[0..hd], &pos1_orig, TOL, "rotate dim64");
    }

    #[test]
    fn test_rotate_dim128() {
        let (heads, max_seq, hd) = (1, 4, 128);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let pos2_orig: Vec<f32> = cache[2 * hd..3 * hd].to_vec();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 2);
        assert_slices_approx(&cache[0..hd], &pos2_orig, TOL, "rotate dim128");
    }

    #[test]
    fn test_rotate_preserves_all_data() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 1.0);
        let mut orig_sorted: Vec<f32> = cache.clone();
        orig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 3);
        let mut rotated_sorted: Vec<f32> = cache.clone();
        rotated_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(orig_sorted, rotated_sorted);
    }

    #[test]
    fn test_rotate_4_heads_shift2() {
        let (heads, max_seq, hd) = (4, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let hs = max_seq * hd;
        let h2_pos3: Vec<f32> = cache[2 * hs + 3 * hd..2 * hs + 4 * hd].to_vec();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 2);
        // head2, pos1 should now be what was at pos3
        assert_slices_approx(&cache[2 * hs + 1 * hd..2 * hs + 2 * hd], &h2_pos3, TOL, "4head rot");
    }

    #[test]
    fn test_rotate_single_position_cache() {
        let (heads, max_seq, hd) = (1, 1, 4);
        let mut cache = vec![1.0, 2.0, 3.0, 4.0];
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        assert_eq!(cache, vec![1.0, 2.0, 3.0, 4.0]); // shift wraps back
    }

    #[test]
    fn test_rotate_two_positions_shift1() {
        let (heads, max_seq, hd) = (1, 2, 4);
        let mut cache = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        assert_eq!(&cache[0..4], &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(&cache[4..8], &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_rotate_large_cache() {
        let (heads, max_seq, hd) = (2, 64, 32);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        // Rotating by max_seq should be identity
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, max_seq);
        assert_slices_approx(&cache, &orig, TOL, "large rotate identity");
    }

    #[test]
    fn test_rotate_inverse() {
        let (heads, max_seq, hd) = (1, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 5);
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 3); // 5 + 3 = 8 = max_seq
        assert_slices_approx(&cache, &orig, TOL, "rotate inverse");
    }

    // ================================================================
    // Concat operations (10+ tests)
    // ================================================================

    #[test]
    fn test_concat_basic() {
        let (heads, hd) = (1, 4);
        let existing = vec![1.0, 2.0, 3.0, 4.0];
        let new_data = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, 1, 1, hd, heads);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_multi_head() {
        let (heads, hd) = (2, 4);
        let ex_len = 2;
        let new_len = 1;
        let existing = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // head 0
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // head 1
        ];
        let new_data = vec![
            20.0, 21.0, 22.0, 23.0, // head 0
            30.0, 31.0, 32.0, 33.0, // head 1
        ];
        let total_seq = ex_len + new_len;
        let mut output = vec![0.0f32; heads * total_seq * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        // head0: existing[0..8] then new[0..4]
        assert_eq!(&output[0..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&output[8..12], &[20.0, 21.0, 22.0, 23.0]);
        // head1
        let h1_base = total_seq * hd;
        assert_eq!(&output[h1_base..h1_base + 8], &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        assert_eq!(&output[h1_base + 8..h1_base + 12], &[30.0, 31.0, 32.0, 33.0]);
    }

    #[test]
    fn test_concat_empty_existing() {
        let (heads, hd) = (1, 4);
        let existing: Vec<f32> = vec![];
        let new_data = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, 0, 1, hd, heads);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_empty_new() {
        let (heads, hd) = (1, 4);
        let existing = vec![1.0, 2.0, 3.0, 4.0];
        let new_data: Vec<f32> = vec![];
        let mut output = vec![0.0f32; 4];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, 1, 0, hd, heads);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concat_preserves_order() {
        let (heads, hd) = (1, 4);
        let existing = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let new_data = vec![90.0, 100.0, 110.0, 120.0];
        let mut output = vec![0.0f32; 12];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, 2, 1, hd, heads);
        assert_eq!(&output[0..8], &existing[..]);
        assert_eq!(&output[8..12], &new_data[..]);
    }

    #[test]
    fn test_concat_dim64() {
        let (heads, hd) = (1, 64);
        let ex_len = 2;
        let new_len = 1;
        let existing: Vec<f32> = (0..ex_len * hd).map(|i| i as f32).collect();
        let new_data: Vec<f32> = (0..new_len * hd).map(|i| (i + 1000) as f32).collect();
        let mut output = vec![0.0f32; (ex_len + new_len) * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        assert_slices_approx(&output[0..ex_len * hd], &existing, TOL, "concat dim64 existing");
        assert_slices_approx(&output[ex_len * hd..], &new_data, TOL, "concat dim64 new");
    }

    #[test]
    fn test_concat_dim128_4heads() {
        let (heads, hd) = (4, 128);
        let ex_len = 3;
        let new_len = 2;
        let existing: Vec<f32> = (0..heads * ex_len * hd).map(|i| (i as f32) * 0.001).collect();
        let new_data: Vec<f32> = (0..heads * new_len * hd).map(|i| (i as f32) * 0.002).collect();
        let total = ex_len + new_len;
        let mut output = vec![0.0f32; heads * total * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        // Check head 0 existing part
        assert_slices_approx(
            &output[0..ex_len * hd],
            &existing[0..ex_len * hd],
            TOL,
            "concat 4h128 h0",
        );
    }

    #[test]
    fn test_concat_negative_values() {
        let (heads, hd) = (1, 4);
        let existing = vec![-1.0, -2.0, -3.0, -4.0];
        let new_data = vec![-5.0, -6.0, -7.0, -8.0];
        let mut output = vec![0.0f32; 8];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, 1, 1, hd, heads);
        assert_eq!(output, vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
    }

    #[test]
    fn test_concat_large_seq() {
        let (heads, hd) = (1, 32);
        let ex_len = 64;
        let new_len = 16;
        let existing: Vec<f32> = (0..ex_len * hd).map(|i| i as f32).collect();
        let new_data: Vec<f32> = (0..new_len * hd).map(|i| (i + 10000) as f32).collect();
        let mut output = vec![0.0f32; (ex_len + new_len) * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[ex_len * hd], 10000.0);
    }

    #[test]
    fn test_concat_single_element_each() {
        let (heads, hd) = (1, 1);
        let existing = vec![42.0];
        let new_data = vec![99.0];
        let mut output = vec![0.0f32; 2];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, 1, 1, hd, heads);
        assert_eq!(output, vec![42.0, 99.0]);
    }

    // ================================================================
    // Quantize/dequantize round-trip (15+ tests)
    // ================================================================

    fn quant_roundtrip(src: &[f32], seq_len: usize, head_dim: usize, num_heads: usize) -> Vec<f32> {
        let total = num_heads * seq_len * head_dim;
        let mut quant = vec![0i8; total];
        let mut scales = vec![0.0f32; num_heads * seq_len];
        let mut restored = vec![0.0f32; total];
        scalar_kv_cache_quantize_store(src, &mut quant, &mut scales, seq_len, head_dim, num_heads);
        scalar_kv_cache_dequantize_load(
            &quant,
            &scales,
            &mut restored,
            seq_len,
            head_dim,
            num_heads,
        );
        restored
    }

    #[test]
    fn test_quant_roundtrip_zeros() {
        let src = vec![0.0f32; 16];
        let restored = quant_roundtrip(&src, 4, 4, 1);
        for v in &restored {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_quant_roundtrip_ones() {
        let src = vec![1.0f32; 16];
        let restored = quant_roundtrip(&src, 4, 4, 1);
        for (i, &v) in restored.iter().enumerate() {
            assert_approx(v, 1.0, QUANT_TOL, &format!("ones[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_negative() {
        let src = vec![-1.0f32; 16];
        let restored = quant_roundtrip(&src, 4, 4, 1);
        for (i, &v) in restored.iter().enumerate() {
            assert_approx(v, -1.0, QUANT_TOL, &format!("neg[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_mixed() {
        let src = vec![1.0, -1.0, 0.5, -0.5, 0.0, 2.0, -2.0, 0.25];
        let restored = quant_roundtrip(&src, 2, 4, 1);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("mixed[{i}]"));
        }
    }

    #[test]
    fn test_quant_preserves_sign() {
        let src = vec![1.0, -1.0, 0.1, -0.1];
        let restored = quant_roundtrip(&src, 1, 4, 1);
        assert!(restored[0] > 0.0, "positive preserved");
        assert!(restored[1] < 0.0, "negative preserved");
        assert!(restored[2] > 0.0, "small positive preserved");
        assert!(restored[3] < 0.0, "small negative preserved");
    }

    #[test]
    fn test_quant_roundtrip_large_values() {
        let src = vec![100.0, -100.0, 50.0, -50.0];
        let restored = quant_roundtrip(&src, 1, 4, 1);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            let rel_err = (rest - orig).abs() / orig.abs().max(1.0);
            assert!(rel_err < 0.02, "large val[{i}]: orig={orig} rest={rest} rel_err={rel_err}");
        }
    }

    #[test]
    fn test_quant_roundtrip_uniform() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let restored = quant_roundtrip(&src, 1, 64, 1);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("uniform[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_multi_head() {
        let (heads, seq, hd) = (4, 2, 8);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("mh quant[{i}]"));
        }
    }

    #[test]
    fn test_quant_scale_correctness() {
        let src = vec![0.0, 0.5, 1.0, 2.0]; // amax = 2.0
        let mut quant = vec![0i8; 4];
        let mut scales = vec![0.0f32; 1];
        scalar_kv_cache_quantize_store(&src, &mut quant, &mut scales, 1, 4, 1);
        let expected_scale = 2.0 / 127.0;
        assert_approx(scales[0], expected_scale, 1e-6, "scale");
    }

    #[test]
    fn test_quant_zero_scale_handling() {
        let src = vec![0.0f32; 8];
        let mut quant = vec![0i8; 8];
        let mut scales = vec![0.0f32; 2];
        scalar_kv_cache_quantize_store(&src, &mut quant, &mut scales, 2, 4, 1);
        assert_eq!(scales[0], 1.0); // zero input → scale=1.0
        assert_eq!(scales[1], 1.0);
    }

    #[test]
    fn test_quant_i8_range() {
        // Values that should map to full i8 range
        let src = vec![127.0, -128.0, 0.0, 64.0];
        let mut quant = vec![0i8; 4];
        let mut scales = vec![0.0f32; 1];
        scalar_kv_cache_quantize_store(&src, &mut quant, &mut scales, 1, 4, 1);
        // amax = 128.0, scale = 128/127, quantized 127 should map to ~127
        assert!(quant[0] == 127 || quant[0] == 126);
        assert!(quant[1] == -128 || quant[1] == -127);
    }

    #[test]
    fn test_quant_roundtrip_dim32() {
        let (heads, seq, hd) = (2, 4, 32);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| ((i as f32) * 0.37).sin()).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("dim32[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_dim64() {
        let (heads, seq, hd) = (1, 2, 64);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| ((i as f32) * 0.23).cos()).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("dim64[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_dim128() {
        let (heads, seq, hd) = (1, 2, 128);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| ((i as f32) * 0.11).sin()).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("dim128[{i}]"));
        }
    }

    #[test]
    fn test_quant_dequant_independent_tokens() {
        // Two tokens with very different magnitudes
        let src = vec![
            0.001, 0.002, 0.003, 0.004, // token 0: tiny
            100.0, 200.0, 300.0, 400.0, // token 1: huge
        ];
        let restored = quant_roundtrip(&src, 2, 4, 1);
        // Each token quantized independently, so tiny values should be preserved well
        for (i, (&orig, &rest)) in src[0..4].iter().zip(restored[0..4].iter()).enumerate() {
            assert_approx(rest, orig, 0.01, &format!("tiny tok[{i}]"));
        }
        for (i, (&orig, &rest)) in src[4..8].iter().zip(restored[4..8].iter()).enumerate() {
            let rel = (rest - orig).abs() / orig.abs();
            assert!(rel < 0.02, "huge tok[{i}]: rel={rel}");
        }
    }

    // ================================================================
    // Various head dimensions (32, 64, 128) (10+ tests)
    // ================================================================

    #[test]
    fn test_append_gather_dim32() {
        let (heads, max_seq, hd) = (2, 8, 32);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = make_new_data(2, heads, hd);
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        let positions = [0u32, 1];
        let mut gathered = vec![0.0f32; 2 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut gathered, hd, heads);
        // gathered should match new_data
        assert_slices_approx(&gathered, &new_data, TOL, "append+gather dim32");
    }

    #[test]
    fn test_append_gather_dim64() {
        let (heads, max_seq, hd) = (2, 8, 64);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = make_new_data(1, heads, hd);
        scalar_kv_cache_append(&mut cache, &new_data, 3, hd, heads);
        let positions = [3u32];
        let mut gathered = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut gathered, hd, heads);
        assert_slices_approx(&gathered, &new_data, TOL, "append+gather dim64");
    }

    #[test]
    fn test_append_gather_dim128() {
        let (heads, max_seq, hd) = (2, 8, 128);
        let mut cache = make_cache(heads, max_seq, hd);
        let new_data = make_new_data(1, heads, hd);
        scalar_kv_cache_append(&mut cache, &new_data, 0, hd, heads);
        let positions = [0u32];
        let mut gathered = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut gathered, hd, heads);
        assert_slices_approx(&gathered, &new_data, TOL, "append+gather dim128");
    }

    #[test]
    fn test_rotate_dim32_shift3() {
        let (heads, max_seq, hd) = (2, 8, 32);
        let mut cache = make_cache(heads, max_seq, hd);
        fill_sequential(&mut cache, 0.0);
        let orig = cache.clone();
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 3);
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 5);
        assert_slices_approx(&cache, &orig, TOL, "rotate dim32 inverse");
    }

    #[test]
    fn test_concat_dim32() {
        let (heads, hd) = (2, 32);
        let ex_len = 3;
        let new_len = 2;
        let existing: Vec<f32> = (0..heads * ex_len * hd).map(|i| i as f32).collect();
        let new_data: Vec<f32> = (0..heads * new_len * hd).map(|i| (i + 5000) as f32).collect();
        let mut output = vec![0.0f32; heads * (ex_len + new_len) * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[ex_len * hd], 5000.0);
    }

    #[test]
    fn test_concat_dim64_single_head() {
        let (heads, hd) = (1, 64);
        let ex_len = 4;
        let new_len = 1;
        let existing: Vec<f32> = (0..ex_len * hd).map(|i| i as f32).collect();
        let new_data: Vec<f32> = (0..new_len * hd).map(|i| (i + 9000) as f32).collect();
        let mut output = vec![0.0f32; (ex_len + new_len) * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        assert_slices_approx(&output[0..ex_len * hd], &existing, TOL, "concat d64 ex");
        assert_slices_approx(&output[ex_len * hd..], &new_data, TOL, "concat d64 new");
    }

    #[test]
    fn test_concat_dim128_multi_head() {
        let (heads, hd) = (4, 128);
        let ex_len = 2;
        let new_len = 1;
        let existing: Vec<f32> = (0..heads * ex_len * hd).map(|i| (i as f32) * 0.001).collect();
        let new_data: Vec<f32> = (0..heads * new_len * hd).map(|i| (i as f32) * -0.001).collect();
        let total = ex_len + new_len;
        let mut output = vec![0.0f32; heads * total * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        // Just verify head 0 existing portion
        assert_slices_approx(
            &output[0..ex_len * hd],
            &existing[0..ex_len * hd],
            TOL,
            "concat d128 mh",
        );
    }

    #[test]
    fn test_quant_roundtrip_8heads_dim32() {
        let (heads, seq, hd) = (8, 4, 32);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| ((i as f32) * 0.07).sin()).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("8h dim32[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_4heads_dim64() {
        let (heads, seq, hd) = (4, 3, 64);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| ((i as f32) * 0.13).cos()).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("4h dim64[{i}]"));
        }
    }

    #[test]
    fn test_quant_roundtrip_2heads_dim128() {
        let (heads, seq, hd) = (2, 2, 128);
        let src: Vec<f32> = (0..heads * seq * hd).map(|i| ((i as f32) * 0.03).sin()).collect();
        let restored = quant_roundtrip(&src, seq, hd, heads);
        for (i, (&orig, &rest)) in src.iter().zip(restored.iter()).enumerate() {
            assert_approx(rest, orig, QUANT_TOL, &format!("2h dim128[{i}]"));
        }
    }

    // ================================================================
    // Edge cases: single token, full cache, empty cache (10+ tests)
    // ================================================================

    #[test]
    fn test_edge_single_token_full_pipeline() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        scalar_kv_cache_append(&mut cache, &data, 0, hd, heads);
        let positions = [0u32];
        let mut gathered = vec![0.0f32; hd];
        scalar_kv_cache_gather(&cache, &positions, &mut gathered, hd, heads);
        assert_eq!(gathered, data);
    }

    #[test]
    fn test_edge_full_cache_append() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        for pos in 0..max_seq {
            let data: Vec<f32> = vec![(pos + 1) as f32; hd];
            scalar_kv_cache_append(&mut cache, &data, pos, hd, heads);
        }
        // All positions filled
        for pos in 0..max_seq {
            let expected = (pos + 1) as f32;
            for d in 0..hd {
                assert_eq!(cache[pos * hd + d], expected);
            }
        }
    }

    #[test]
    fn test_edge_empty_cache_gather() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let cache = make_cache(heads, max_seq, hd); // all zeros
        let positions = [0u32, 1, 2];
        let mut output = vec![99.0f32; 3 * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut output, hd, heads);
        // Should read zeros
        for v in &output {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_edge_single_head_dim_1() {
        let (heads, max_seq, hd) = (1, 8, 1);
        let mut cache = vec![0.0f32; max_seq];
        let data = vec![42.0];
        scalar_kv_cache_append(&mut cache, &data, 5, hd, heads);
        assert_eq!(cache[5], 42.0);
    }

    #[test]
    fn test_edge_rotate_full_cache() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        for pos in 0..max_seq {
            let data = vec![(pos + 1) as f32; hd];
            scalar_kv_cache_append(&mut cache, &data, pos, hd, heads);
        }
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        // pos0 should now have what was at pos1
        assert_eq!(cache[0], 2.0);
        assert_eq!(cache[hd * 3], 1.0); // last pos gets wrapped pos0
    }

    #[test]
    fn test_edge_gather_single_head_single_pos() {
        let cache = vec![10.0, 20.0, 30.0, 40.0];
        let positions = [0u32];
        let mut output = vec![0.0f32; 4];
        scalar_kv_cache_gather(&cache, &positions, &mut output, 4, 1);
        assert_eq!(output, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_edge_quant_single_value() {
        let src = vec![3.14];
        let restored = quant_roundtrip(&src, 1, 1, 1);
        assert_approx(restored[0], 3.14, QUANT_TOL, "single quant");
    }

    #[test]
    fn test_edge_quant_all_same_value() {
        let src = vec![0.5f32; 32];
        let restored = quant_roundtrip(&src, 1, 32, 1);
        for (i, &v) in restored.iter().enumerate() {
            assert_approx(v, 0.5, QUANT_TOL, &format!("same val[{i}]"));
        }
    }

    #[test]
    fn test_edge_concat_many_tokens() {
        let (heads, hd) = (1, 4);
        let ex_len = 32;
        let new_len = 32;
        let existing: Vec<f32> = vec![1.0; ex_len * hd];
        let new_data: Vec<f32> = vec![2.0; new_len * hd];
        let mut output = vec![0.0f32; (ex_len + new_len) * hd];
        scalar_kv_cache_concat(&existing, &new_data, &mut output, ex_len, new_len, hd, heads);
        assert!(output[0..ex_len * hd].iter().all(|&v| v == 1.0));
        assert!(output[ex_len * hd..].iter().all(|&v| v == 2.0));
    }

    #[test]
    fn test_edge_full_pipeline_multi_step() {
        let (heads, max_seq, hd) = (2, 8, 4);
        let mut cache = make_cache(heads, max_seq, hd);

        // Step 1: append 3 tokens
        let data3 = make_new_data(3, heads, hd);
        scalar_kv_cache_append(&mut cache, &data3, 0, hd, heads);

        // Step 2: gather positions 0 and 2
        let positions = [0u32, 2];
        let mut gathered = vec![0.0f32; 2 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions, &mut gathered, hd, heads);

        // Step 3: append one more token at position 3
        let data1 = make_new_data(1, heads, hd);
        scalar_kv_cache_append(&mut cache, &data1, 3, hd, heads);

        // Step 4: gather the new token
        let positions2 = [3u32];
        let mut gathered2 = vec![0.0f32; 1 * heads * hd];
        scalar_kv_cache_gather(&cache, &positions2, &mut gathered2, hd, heads);
        assert_slices_approx(&gathered2, &data1, TOL, "pipeline step4");
    }

    #[test]
    fn test_edge_rotate_then_append() {
        let (heads, max_seq, hd) = (1, 4, 4);
        let mut cache = make_cache(heads, max_seq, hd);
        // Fill all positions
        for pos in 0..max_seq {
            let data = vec![(pos + 1) as f32; hd];
            scalar_kv_cache_append(&mut cache, &data, pos, hd, heads);
        }
        // Rotate by 1 to evict oldest
        scalar_kv_cache_rotate(&mut cache, max_seq, hd, heads, 1);
        // Now write new data to the last position (which wrapped around)
        let new_tok = vec![99.0; hd];
        scalar_kv_cache_append(&mut cache, &new_tok, max_seq - 1, hd, heads);
        assert_eq!(cache[(max_seq - 1) * hd], 99.0);
    }

    #[test]
    fn test_edge_quant_with_inf() {
        let src = vec![f32::INFINITY, 0.0, -f32::INFINITY, 1.0];
        let mut quant = vec![0i8; 4];
        let mut scales = vec![0.0f32; 1];
        // Should not panic — inf produces inf scale but no crash
        scalar_kv_cache_quantize_store(&src, &mut quant, &mut scales, 1, 4, 1);
        // Scale will be inf/127 = inf
        assert!(scales[0].is_infinite() || scales[0].is_nan() || scales[0] > 1e30);
    }
}
