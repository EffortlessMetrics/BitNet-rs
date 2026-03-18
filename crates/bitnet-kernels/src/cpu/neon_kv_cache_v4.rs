#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]
//! NEON-optimized KV cache v4 operations for Apple Silicon.
//!
//! Provides advanced KV cache operations — append, gather, rotate,
//! quantize, and dequantize — using NEON SIMD intrinsics on AArch64
//! with scalar fallbacks for portability.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar fallbacks ────────────────────────────────────────────────

/// Scalar fallback: append new KV pairs into `cache` at `pos`.
pub fn kv_cache_append_scalar(
    cache: &mut [f32],
    new_kv: &[f32],
    pos: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let stride = head_dim * num_heads;
    assert!(new_kv.len() >= stride, "new_kv too small: need {stride}, got {}", new_kv.len());
    let dst_offset = pos * stride;
    assert!(
        cache.len() >= dst_offset + stride,
        "cache too small: need {}, got {}",
        dst_offset + stride,
        cache.len()
    );
    cache[dst_offset..dst_offset + stride].copy_from_slice(&new_kv[..stride]);
}

/// Scalar fallback: gather positions from cache.
pub fn kv_cache_gather_scalar(
    cache: &[f32],
    indices: &[usize],
    output: &mut [f32],
    head_dim: usize,
) {
    let out_len = indices.len() * head_dim;
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());
    for (i, &idx) in indices.iter().enumerate() {
        let src_off = idx * head_dim;
        assert!(
            cache.len() >= src_off + head_dim,
            "cache index out of bounds: idx={idx} head_dim={head_dim} cache.len={}",
            cache.len()
        );
        let dst_off = i * head_dim;
        output[dst_off..dst_off + head_dim].copy_from_slice(&cache[src_off..src_off + head_dim]);
    }
}

/// Scalar fallback: circular-buffer rotation.
pub fn kv_cache_rotate_scalar(
    cache: &mut [f32],
    positions: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let stride = head_dim * num_heads;
    let total = cache.len();
    if total == 0 || stride == 0 || positions == 0 {
        return;
    }
    let num_positions = total / stride;
    if num_positions <= 1 {
        return;
    }
    let rotate_by = positions % num_positions;
    if rotate_by == 0 {
        return;
    }
    // Rotate left by `rotate_by` positions using the three-reverse trick.
    let byte_pivot = rotate_by * stride;
    cache[..byte_pivot].reverse();
    cache[byte_pivot..num_positions * stride].reverse();
    cache[..num_positions * stride].reverse();
}

/// Scalar fallback: quantize f32 cache to i8 with per-position scale.
pub fn kv_cache_quantize_scalar(
    cache: &[f32],
    output: &mut [i8],
    scale: &mut [f32],
    head_dim: usize,
    num_positions: usize,
) {
    assert!(cache.len() >= num_positions * head_dim, "cache too small for quantization");
    assert!(output.len() >= num_positions * head_dim, "output too small for quantization");
    assert!(
        scale.len() >= num_positions,
        "scale too small: need {num_positions}, got {}",
        scale.len()
    );
    for pos in 0..num_positions {
        let off = pos * head_dim;
        let row = &cache[off..off + head_dim];
        let abs_max = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let s = if abs_max == 0.0 { 1.0 } else { 127.0 / abs_max };
        scale[pos] = abs_max / 127.0;
        for j in 0..head_dim {
            let q = (row[j] * s).round().clamp(-128.0, 127.0) as i8;
            output[off + j] = q;
        }
    }
}

/// Scalar fallback: dequantize i8 cache back to f32.
pub fn kv_cache_dequantize_scalar(
    cache: &[i8],
    scale: &[f32],
    output: &mut [f32],
    head_dim: usize,
    num_positions: usize,
) {
    assert!(cache.len() >= num_positions * head_dim, "cache too small for dequantization");
    assert!(output.len() >= num_positions * head_dim, "output too small for dequantization");
    assert!(
        scale.len() >= num_positions,
        "scale too small: need {num_positions}, got {}",
        scale.len()
    );
    for pos in 0..num_positions {
        let off = pos * head_dim;
        let s = scale[pos];
        for j in 0..head_dim {
            output[off + j] = cache[off + j] as f32 * s;
        }
    }
}

// ── NEON-optimized implementations ──────────────────────────────────

/// Append new KV pairs into `cache` at position `pos` using NEON copy.
///
/// `cache` is laid out as `[max_positions, num_heads * head_dim]` in
/// row-major order. `new_kv` must contain at least `num_heads * head_dim`
/// elements.
#[cfg(target_arch = "aarch64")]
pub fn kv_cache_append_neon(
    cache: &mut [f32],
    new_kv: &[f32],
    pos: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let stride = head_dim * num_heads;
    assert!(new_kv.len() >= stride, "new_kv too small: need {stride}, got {}", new_kv.len());
    let dst_offset = pos * stride;
    assert!(
        cache.len() >= dst_offset + stride,
        "cache too small: need {}, got {}",
        dst_offset + stride,
        cache.len()
    );

    let chunks = stride / 4;
    let remainder = stride % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(new_kv.as_ptr().add(base));
            vst1q_f32(cache.as_mut_ptr().add(dst_offset + base), v);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        cache[dst_offset + tail + i] = new_kv[tail + i];
    }
}

/// Gather specific positions from cache using NEON-accelerated copy.
///
/// For each index in `indices`, copies `head_dim` elements from the
/// corresponding cache row into contiguous `output` storage.
#[cfg(target_arch = "aarch64")]
pub fn kv_cache_gather_neon(cache: &[f32], indices: &[usize], output: &mut [f32], head_dim: usize) {
    let out_len = indices.len() * head_dim;
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    let chunks = head_dim / 4;
    let remainder = head_dim % 4;

    for (i, &idx) in indices.iter().enumerate() {
        let src_off = idx * head_dim;
        assert!(
            cache.len() >= src_off + head_dim,
            "cache index out of bounds: idx={idx} head_dim={head_dim} cache.len={}",
            cache.len()
        );
        let dst_off = i * head_dim;

        for c in 0..chunks {
            let base = c * 4;
            unsafe {
                let v = vld1q_f32(cache.as_ptr().add(src_off + base));
                vst1q_f32(output.as_mut_ptr().add(dst_off + base), v);
            }
        }
        let tail = chunks * 4;
        for j in 0..remainder {
            output[dst_off + tail + j] = cache[src_off + tail + j];
        }
    }
}

/// Circular-buffer rotation using NEON-accelerated block moves.
///
/// Rotates the cache left by `positions` slots so that the oldest
/// entries are evicted, using the three-reverse algorithm with NEON
/// loads/stores for the reversal passes.
#[cfg(target_arch = "aarch64")]
pub fn kv_cache_rotate_neon(
    cache: &mut [f32],
    positions: usize,
    head_dim: usize,
    num_heads: usize,
) {
    let stride = head_dim * num_heads;
    let total = cache.len();
    if total == 0 || stride == 0 || positions == 0 {
        return;
    }
    let num_positions = total / stride;
    if num_positions <= 1 {
        return;
    }
    let rotate_by = positions % num_positions;
    if rotate_by == 0 {
        return;
    }
    let elem_count = num_positions * stride;
    let pivot = rotate_by * stride;

    neon_reverse_f32(&mut cache[..pivot]);
    neon_reverse_f32(&mut cache[pivot..elem_count]);
    neon_reverse_f32(&mut cache[..elem_count]);
}

/// NEON-accelerated in-place reverse of an f32 slice.
#[cfg(target_arch = "aarch64")]
fn neon_reverse_f32(data: &mut [f32]) {
    let len = data.len();
    if len <= 1 {
        return;
    }
    let mut lo = 0usize;
    let mut hi = len;
    // Swap 4-element chunks from both ends using NEON.
    while lo + 4 <= hi.saturating_sub(4) {
        hi -= 4;
        unsafe {
            let a = vld1q_f32(data.as_ptr().add(lo));
            let b = vld1q_f32(data.as_ptr().add(hi));
            // Reverse each 4-lane vector before swapping.
            let a_rev = vrev64q_f32(a);
            let a_rev = vextq_f32(a_rev, a_rev, 2);
            let b_rev = vrev64q_f32(b);
            let b_rev = vextq_f32(b_rev, b_rev, 2);
            vst1q_f32(data.as_mut_ptr().add(lo), b_rev);
            vst1q_f32(data.as_mut_ptr().add(hi), a_rev);
        }
        lo += 4;
    }
    // Scalar tail.
    while lo < hi.saturating_sub(1) {
        hi -= 1;
        data.swap(lo, hi);
        lo += 1;
    }
}

/// Quantize f32 cache to int8 with per-position absmax scaling.
///
/// For each of `num_positions` rows of length `head_dim`, computes the
/// absolute-maximum, stores the scale factor (`absmax / 127`) in `scale`,
/// and writes the quantized int8 values to `output`.
#[cfg(target_arch = "aarch64")]
pub fn kv_cache_quantize_neon(
    cache: &[f32],
    output: &mut [i8],
    scale: &mut [f32],
    head_dim: usize,
    num_positions: usize,
) {
    assert!(cache.len() >= num_positions * head_dim, "cache too small for quantization");
    assert!(output.len() >= num_positions * head_dim, "output too small for quantization");
    assert!(
        scale.len() >= num_positions,
        "scale too small: need {num_positions}, got {}",
        scale.len()
    );

    for pos in 0..num_positions {
        let off = pos * head_dim;
        let row = &cache[off..off + head_dim];

        // --- absmax via NEON ---
        let chunks = head_dim / 4;
        let remainder = head_dim % 4;
        let mut vmax = unsafe { vdupq_n_f32(0.0) };
        for c in 0..chunks {
            let base = c * 4;
            unsafe {
                let v = vld1q_f32(row.as_ptr().add(base));
                let a = vabsq_f32(v);
                vmax = vmaxq_f32(vmax, a);
            }
        }
        let mut abs_max: f32 = unsafe { vmaxvq_f32(vmax) };
        let tail = chunks * 4;
        for i in 0..remainder {
            abs_max = abs_max.max(row[tail + i].abs());
        }

        let s = if abs_max == 0.0 { 1.0 } else { 127.0 / abs_max };
        scale[pos] = abs_max / 127.0;

        // --- quantize via NEON ---
        let scale_v = unsafe { vdupq_n_f32(s) };
        for c in 0..chunks {
            let base = c * 4;
            unsafe {
                let v = vld1q_f32(row.as_ptr().add(base));
                let scaled = vmulq_f32(v, scale_v);
                // Convert to i32, narrow to i16, narrow to i8.
                let i32v = vcvtnq_s32_f32(scaled);
                let i16v = vqmovn_s32(i32v);
                // We need 8 lanes for vqmovn_s16; duplicate the 4 lanes.
                let i16_full = vcombine_s16(i16v, i16v);
                let i8v = vqmovn_s16(i16_full);
                // Store only lower 4 bytes.
                let out_ptr = output.as_mut_ptr().add(off + base) as *mut u8;
                vst1_lane_u8::<0>(out_ptr, vreinterpret_u8_s8(i8v));
                vst1_lane_u8::<1>(out_ptr.add(1), vreinterpret_u8_s8(i8v));
                vst1_lane_u8::<2>(out_ptr.add(2), vreinterpret_u8_s8(i8v));
                vst1_lane_u8::<3>(out_ptr.add(3), vreinterpret_u8_s8(i8v));
            }
        }
        for i in 0..remainder {
            let q = (row[tail + i] * s).round().clamp(-128.0, 127.0) as i8;
            output[off + tail + i] = q;
        }
    }
}

/// Dequantize int8 cache back to f32 using per-position scale.
#[cfg(target_arch = "aarch64")]
pub fn kv_cache_dequantize_neon(
    cache: &[i8],
    scale: &[f32],
    output: &mut [f32],
    head_dim: usize,
    num_positions: usize,
) {
    assert!(cache.len() >= num_positions * head_dim, "cache too small for dequantization");
    assert!(output.len() >= num_positions * head_dim, "output too small for dequantization");
    assert!(
        scale.len() >= num_positions,
        "scale too small: need {num_positions}, got {}",
        scale.len()
    );

    for pos in 0..num_positions {
        let off = pos * head_dim;
        let s = scale[pos];
        let scale_v = unsafe { vdupq_n_f32(s) };

        let chunks = head_dim / 4;
        let remainder = head_dim % 4;

        for c in 0..chunks {
            let base = c * 4;
            unsafe {
                // Load 4 × i8, widen to i32, convert to f32.
                let in_ptr = cache.as_ptr().add(off + base);
                let b0 = *in_ptr as i32;
                let b1 = *in_ptr.add(1) as i32;
                let b2 = *in_ptr.add(2) as i32;
                let b3 = *in_ptr.add(3) as i32;
                let i32v = vld1q_s32([b0, b1, b2, b3].as_ptr());
                let fv = vcvtq_f32_s32(i32v);
                let result = vmulq_f32(fv, scale_v);
                vst1q_f32(output.as_mut_ptr().add(off + base), result);
            }
        }
        let tail = chunks * 4;
        for i in 0..remainder {
            output[off + tail + i] = cache[off + tail + i] as f32 * s;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a cache buffer of given shape.
    fn make_cache(max_pos: usize, head_dim: usize, num_heads: usize) -> Vec<f32> {
        vec![0.0f32; max_pos * head_dim * num_heads]
    }

    fn make_kv_row(head_dim: usize, num_heads: usize, base: f32) -> Vec<f32> {
        (0..head_dim * num_heads).map(|i| base + i as f32 * 0.1).collect()
    }

    fn sequential_f32(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32 + 1.0).collect()
    }

    // ── Append tests ────────────────────────────────────────────────

    #[test]
    fn test_append_scalar_basic() {
        let mut cache = make_cache(4, 32, 1);
        let kv = make_kv_row(32, 1, 1.0);
        kv_cache_append_scalar(&mut cache, &kv, 0, 32, 1);
        assert_eq!(&cache[..32], &kv[..32]);
    }

    #[test]
    fn test_append_scalar_second_position() {
        let mut cache = make_cache(4, 64, 2);
        let kv = make_kv_row(64, 2, 5.0);
        kv_cache_append_scalar(&mut cache, &kv, 1, 64, 2);
        let stride = 64 * 2;
        assert_eq!(&cache[stride..stride * 2], &kv[..stride]);
    }

    #[test]
    fn test_append_scalar_multi_head() {
        let mut cache = make_cache(8, 32, 4);
        let kv = make_kv_row(32, 4, 0.0);
        kv_cache_append_scalar(&mut cache, &kv, 3, 32, 4);
        let stride = 32 * 4;
        assert_eq!(&cache[3 * stride..4 * stride], &kv[..stride]);
    }

    #[test]
    #[should_panic(expected = "cache too small")]
    fn test_append_scalar_out_of_bounds() {
        let mut cache = make_cache(2, 32, 1);
        let kv = make_kv_row(32, 1, 1.0);
        kv_cache_append_scalar(&mut cache, &kv, 5, 32, 1);
    }

    #[test]
    fn test_append_scalar_head_dim_128() {
        let mut cache = make_cache(4, 128, 1);
        let kv = make_kv_row(128, 1, 2.0);
        kv_cache_append_scalar(&mut cache, &kv, 2, 128, 1);
        assert_eq!(&cache[256..384], &kv[..128]);
    }

    // ── Gather tests ────────────────────────────────────────────────

    #[test]
    fn test_gather_scalar_basic() {
        let cache = sequential_f32(4 * 32);
        let indices = vec![0, 2];
        let mut output = vec![0.0f32; 2 * 32];
        kv_cache_gather_scalar(&cache, &indices, &mut output, 32);
        assert_eq!(&output[..32], &cache[..32]);
        assert_eq!(&output[32..64], &cache[64..96]);
    }

    #[test]
    fn test_gather_scalar_single_index() {
        let cache = sequential_f32(8 * 64);
        let indices = [3];
        let mut output = [0.0f32; 64];
        kv_cache_gather_scalar(&cache, &indices, &mut output, 64);
        assert_eq!(&output[..64], &cache[192..256]);
    }

    #[test]
    fn test_gather_scalar_reverse_order() {
        let cache = sequential_f32(4 * 32);
        let indices = vec![3, 2, 1, 0];
        let mut output = vec![0.0f32; 4 * 32];
        kv_cache_gather_scalar(&cache, &indices, &mut output, 32);
        assert_eq!(&output[..32], &cache[96..128]);
        assert_eq!(&output[96..128], &cache[..32]);
    }

    #[test]
    fn test_gather_scalar_empty_indices() {
        let cache = sequential_f32(4 * 32);
        let indices: Vec<usize> = vec![];
        let mut output = [0.0f32; 0];
        kv_cache_gather_scalar(&cache, &indices, &mut output, 32);
        assert!(output.is_empty());
    }

    #[test]
    #[should_panic(expected = "cache index out of bounds")]
    fn test_gather_scalar_out_of_bounds() {
        let cache = sequential_f32(4 * 32);
        let indices = [10];
        let mut output = [0.0f32; 32];
        kv_cache_gather_scalar(&cache, &indices, &mut output, 32);
    }

    #[test]
    fn test_gather_scalar_head_dim_128() {
        let cache = sequential_f32(4 * 128);
        let indices = vec![1, 3];
        let mut output = vec![0.0f32; 2 * 128];
        kv_cache_gather_scalar(&cache, &indices, &mut output, 128);
        assert_eq!(&output[..128], &cache[128..256]);
        assert_eq!(&output[128..256], &cache[384..512]);
    }

    // ── Rotate tests ────────────────────────────────────────────────

    #[test]
    fn test_rotate_scalar_basic() {
        let mut cache: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        kv_cache_rotate_scalar(&mut cache, 1, 1, 1);
        assert_eq!(cache, vec![2.0, 3.0, 4.0, 1.0]);
    }

    #[test]
    fn test_rotate_scalar_two_positions() {
        // 4 positions, head_dim=2, num_heads=1 → stride=2
        let mut cache: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        kv_cache_rotate_scalar(&mut cache, 2, 2, 1);
        assert_eq!(cache, vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rotate_scalar_full_rotation() {
        let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut cache = original.clone();
        kv_cache_rotate_scalar(&mut cache, 4, 1, 1);
        assert_eq!(cache, original);
    }

    #[test]
    fn test_rotate_scalar_zero() {
        let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut cache = original.clone();
        kv_cache_rotate_scalar(&mut cache, 0, 1, 1);
        assert_eq!(cache, original);
    }

    #[test]
    fn test_rotate_scalar_empty() {
        let mut cache: Vec<f32> = vec![];
        kv_cache_rotate_scalar(&mut cache, 1, 1, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_rotate_scalar_single_position() {
        let mut cache: Vec<f32> = vec![42.0, 43.0];
        kv_cache_rotate_scalar(&mut cache, 5, 2, 1);
        assert_eq!(cache, vec![42.0, 43.0]);
    }

    #[test]
    fn test_rotate_scalar_multi_head() {
        // 3 positions, head_dim=2, num_heads=2 → stride=4
        let mut cache: Vec<f32> =
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        kv_cache_rotate_scalar(&mut cache, 1, 2, 2);
        assert_eq!(cache, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]);
    }

    // ── Quantize / dequantize tests ─────────────────────────────────

    #[test]
    fn test_quantize_scalar_basic() {
        let cache = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.0];
        let mut output = [0i8; 8];
        let mut scale = [0.0f32; 1];
        kv_cache_quantize_scalar(&cache, &mut output, &mut scale, 8, 1);
        assert!(scale[0] > 0.0);
        assert_eq!(output[0], 127);
        assert_eq!(output[1], -127);
    }

    #[test]
    fn test_quantize_scalar_zeros() {
        let cache = [0.0f32; 32];
        let mut output = [0i8; 32];
        let mut scale = [0.0f32; 1];
        kv_cache_quantize_scalar(&cache, &mut output, &mut scale, 32, 1);
        assert!(output.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantize_scalar_multi_position() {
        let cache: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut output = [0i8; 64];
        let mut scale = [0.0f32; 2];
        kv_cache_quantize_scalar(&cache, &mut output, &mut scale, 32, 2);
        assert!(scale[0] > 0.0);
        assert!(scale[1] > 0.0);
    }

    #[test]
    fn test_dequantize_scalar_basic() {
        let cache = vec![127i8, -127, 64, -64];
        let scale = vec![1.0 / 127.0];
        let mut output = [0.0f32; 4];
        kv_cache_dequantize_scalar(&cache, &scale, &mut output, 4, 1);
        assert!((output[0] - 1.0).abs() < 0.01);
        assert!((output[1] + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_scalar() {
        let cache: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let mut q_out = [0i8; 128];
        let mut scale = [0.0f32; 1];
        kv_cache_quantize_scalar(&cache, &mut q_out, &mut scale, 128, 1);

        let mut deq_out = [0.0f32; 128];
        kv_cache_dequantize_scalar(&q_out, &scale, &mut deq_out, 128, 1);

        for (orig, deq) in cache.iter().zip(deq_out.iter()) {
            assert!((orig - deq).abs() < 0.02, "round-trip error too large: orig={orig} deq={deq}");
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_scalar_multi() {
        let cache: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.005).collect();
        let mut q_out = [0i8; 256];
        let mut scale = [0.0f32; 4];
        kv_cache_quantize_scalar(&cache, &mut q_out, &mut scale, 64, 4);

        let mut deq_out = [0.0f32; 256];
        kv_cache_dequantize_scalar(&q_out, &scale, &mut deq_out, 64, 4);

        for (orig, deq) in cache.iter().zip(deq_out.iter()) {
            assert!((orig - deq).abs() < 0.02, "round-trip error too large: orig={orig} deq={deq}");
        }
    }

    #[test]
    #[should_panic(expected = "cache too small")]
    fn test_quantize_scalar_cache_too_small() {
        let cache = [1.0f32; 4];
        let mut output = [0i8; 32];
        let mut scale = [0.0f32; 1];
        kv_cache_quantize_scalar(&cache, &mut output, &mut scale, 32, 1);
    }

    #[test]
    #[should_panic(expected = "cache too small")]
    fn test_dequantize_scalar_cache_too_small() {
        let cache = [1i8; 4];
        let scale = [1.0f32; 1];
        let mut output = [0.0f32; 32];
        kv_cache_dequantize_scalar(&cache, &scale, &mut output, 32, 1);
    }

    // ── NEON-specific tests (aarch64 only) ──────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod neon_tests {
        use super::*;

        // ── Append NEON ─────────────────────────────────────────────

        #[test]
        fn test_append_neon_basic() {
            let mut cache = make_cache(4, 32, 1);
            let kv = make_kv_row(32, 1, 1.0);
            kv_cache_append_neon(&mut cache, &kv, 0, 32, 1);
            assert_eq!(&cache[..32], &kv[..32]);
        }

        #[test]
        fn test_append_neon_second_position() {
            let mut cache = make_cache(4, 64, 2);
            let kv = make_kv_row(64, 2, 5.0);
            kv_cache_append_neon(&mut cache, &kv, 1, 64, 2);
            let stride = 64 * 2;
            assert_eq!(&cache[stride..stride * 2], &kv[..stride]);
        }

        #[test]
        fn test_append_neon_multi_head() {
            let mut cache = make_cache(8, 32, 4);
            let kv = make_kv_row(32, 4, 0.0);
            kv_cache_append_neon(&mut cache, &kv, 3, 32, 4);
            let stride = 32 * 4;
            assert_eq!(&cache[3 * stride..4 * stride], &kv[..stride]);
        }

        #[test]
        fn test_append_neon_head_dim_128() {
            let mut cache = make_cache(4, 128, 1);
            let kv = make_kv_row(128, 1, 2.0);
            kv_cache_append_neon(&mut cache, &kv, 2, 128, 1);
            assert_eq!(&cache[256..384], &kv[..128]);
        }

        #[test]
        fn test_append_neon_non_aligned() {
            // head_dim=33 is not a multiple of 4.
            let mut cache = make_cache(2, 33, 1);
            let kv = make_kv_row(33, 1, 1.0);
            kv_cache_append_neon(&mut cache, &kv, 0, 33, 1);
            assert_eq!(&cache[..33], &kv[..33]);
        }

        #[test]
        #[should_panic(expected = "cache too small")]
        fn test_append_neon_out_of_bounds() {
            let mut cache = make_cache(2, 32, 1);
            let kv = make_kv_row(32, 1, 1.0);
            kv_cache_append_neon(&mut cache, &kv, 5, 32, 1);
        }

        // ── Append parity ───────────────────────────────────────────

        #[test]
        fn test_append_parity_head32() {
            let kv = make_kv_row(32, 2, 3.0);
            let mut c_neon = make_cache(4, 32, 2);
            let mut c_scalar = make_cache(4, 32, 2);
            kv_cache_append_neon(&mut c_neon, &kv, 1, 32, 2);
            kv_cache_append_scalar(&mut c_scalar, &kv, 1, 32, 2);
            assert_eq!(c_neon, c_scalar);
        }

        #[test]
        fn test_append_parity_head64() {
            let kv = make_kv_row(64, 1, 7.0);
            let mut c_neon = make_cache(8, 64, 1);
            let mut c_scalar = make_cache(8, 64, 1);
            kv_cache_append_neon(&mut c_neon, &kv, 5, 64, 1);
            kv_cache_append_scalar(&mut c_scalar, &kv, 5, 64, 1);
            assert_eq!(c_neon, c_scalar);
        }

        #[test]
        fn test_append_parity_head128() {
            let kv = make_kv_row(128, 4, 0.5);
            let mut c_neon = make_cache(4, 128, 4);
            let mut c_scalar = make_cache(4, 128, 4);
            kv_cache_append_neon(&mut c_neon, &kv, 2, 128, 4);
            kv_cache_append_scalar(&mut c_scalar, &kv, 2, 128, 4);
            assert_eq!(c_neon, c_scalar);
        }

        // ── Gather NEON ─────────────────────────────────────────────

        #[test]
        fn test_gather_neon_basic() {
            let cache = sequential_f32(4 * 32);
            let indices = vec![0, 2];
            let mut output = vec![0.0f32; 2 * 32];
            kv_cache_gather_neon(&cache, &indices, &mut output, 32);
            assert_eq!(&output[..32], &cache[..32]);
            assert_eq!(&output[32..64], &cache[64..96]);
        }

        #[test]
        fn test_gather_neon_single_index() {
            let cache = sequential_f32(8 * 64);
            let indices = [3];
            let mut output = [0.0f32; 64];
            kv_cache_gather_neon(&cache, &indices, &mut output, 64);
            assert_eq!(&output[..64], &cache[192..256]);
        }

        #[test]
        fn test_gather_neon_reverse_order() {
            let cache = sequential_f32(4 * 32);
            let indices = vec![3, 2, 1, 0];
            let mut output = vec![0.0f32; 4 * 32];
            kv_cache_gather_neon(&cache, &indices, &mut output, 32);
            assert_eq!(&output[..32], &cache[96..128]);
            assert_eq!(&output[96..128], &cache[..32]);
        }

        #[test]
        fn test_gather_neon_empty() {
            let cache = sequential_f32(4 * 32);
            let indices: Vec<usize> = vec![];
            let mut output = [0.0f32; 0];
            kv_cache_gather_neon(&cache, &indices, &mut output, 32);
            assert!(output.is_empty());
        }

        #[test]
        fn test_gather_neon_head_dim_128() {
            let cache = sequential_f32(4 * 128);
            let indices = vec![1, 3];
            let mut output = vec![0.0f32; 2 * 128];
            kv_cache_gather_neon(&cache, &indices, &mut output, 128);
            assert_eq!(&output[..128], &cache[128..256]);
            assert_eq!(&output[128..256], &cache[384..512]);
        }

        #[test]
        #[should_panic(expected = "cache index out of bounds")]
        fn test_gather_neon_out_of_bounds() {
            let cache = sequential_f32(4 * 32);
            let indices = [10];
            let mut output = [0.0f32; 32];
            kv_cache_gather_neon(&cache, &indices, &mut output, 32);
        }

        // ── Gather parity ───────────────────────────────────────────

        #[test]
        fn test_gather_parity_head32() {
            let cache = sequential_f32(8 * 32);
            let indices = vec![1, 3, 5, 7];
            let mut o_neon = vec![0.0f32; 4 * 32];
            let mut o_scalar = vec![0.0f32; 4 * 32];
            kv_cache_gather_neon(&cache, &indices, &mut o_neon, 32);
            kv_cache_gather_scalar(&cache, &indices, &mut o_scalar, 32);
            assert_eq!(o_neon, o_scalar);
        }

        #[test]
        fn test_gather_parity_head64() {
            let cache = sequential_f32(4 * 64);
            let indices = vec![0, 3];
            let mut o_neon = vec![0.0f32; 2 * 64];
            let mut o_scalar = vec![0.0f32; 2 * 64];
            kv_cache_gather_neon(&cache, &indices, &mut o_neon, 64);
            kv_cache_gather_scalar(&cache, &indices, &mut o_scalar, 64);
            assert_eq!(o_neon, o_scalar);
        }

        #[test]
        fn test_gather_parity_head128() {
            let cache = sequential_f32(4 * 128);
            let indices = vec![2, 0, 3, 1];
            let mut o_neon = vec![0.0f32; 4 * 128];
            let mut o_scalar = vec![0.0f32; 4 * 128];
            kv_cache_gather_neon(&cache, &indices, &mut o_neon, 128);
            kv_cache_gather_scalar(&cache, &indices, &mut o_scalar, 128);
            assert_eq!(o_neon, o_scalar);
        }

        // ── Rotate NEON ─────────────────────────────────────────────

        #[test]
        fn test_rotate_neon_basic() {
            let mut cache: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            kv_cache_rotate_neon(&mut cache, 1, 1, 1);
            assert_eq!(cache, vec![2.0, 3.0, 4.0, 1.0]);
        }

        #[test]
        fn test_rotate_neon_two_positions() {
            let mut cache: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            kv_cache_rotate_neon(&mut cache, 2, 2, 1);
            assert_eq!(cache, vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_rotate_neon_full_rotation() {
            let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let mut cache = original.clone();
            kv_cache_rotate_neon(&mut cache, 4, 1, 1);
            assert_eq!(cache, original);
        }

        #[test]
        fn test_rotate_neon_zero() {
            let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let mut cache = original.clone();
            kv_cache_rotate_neon(&mut cache, 0, 1, 1);
            assert_eq!(cache, original);
        }

        #[test]
        fn test_rotate_neon_empty() {
            let mut cache: Vec<f32> = vec![];
            kv_cache_rotate_neon(&mut cache, 1, 1, 1);
            assert!(cache.is_empty());
        }

        #[test]
        fn test_rotate_neon_single_position() {
            let mut cache: Vec<f32> = vec![42.0, 43.0];
            kv_cache_rotate_neon(&mut cache, 5, 2, 1);
            assert_eq!(cache, vec![42.0, 43.0]);
        }

        #[test]
        fn test_rotate_neon_multi_head() {
            let mut cache: Vec<f32> =
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
            kv_cache_rotate_neon(&mut cache, 1, 2, 2);
            assert_eq!(cache, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0]);
        }

        // ── Rotate parity ───────────────────────────────────────────

        #[test]
        fn test_rotate_parity_basic() {
            let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let mut c_neon = data.clone();
            let mut c_scalar = data.clone();
            kv_cache_rotate_neon(&mut c_neon, 3, 2, 1);
            kv_cache_rotate_scalar(&mut c_scalar, 3, 2, 1);
            assert_eq!(c_neon, c_scalar);
        }

        #[test]
        fn test_rotate_parity_large() {
            let data: Vec<f32> = (0..512).map(|i| i as f32 * 0.1).collect();
            let mut c_neon = data.clone();
            let mut c_scalar = data.clone();
            kv_cache_rotate_neon(&mut c_neon, 5, 32, 2);
            kv_cache_rotate_scalar(&mut c_scalar, 5, 32, 2);
            assert_eq!(c_neon, c_scalar);
        }

        // ── Quantize NEON ───────────────────────────────────────────

        #[test]
        fn test_quantize_neon_basic() {
            let cache = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 0.25, -0.25, 0.0];
            let mut output = [0i8; 8];
            let mut scale = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut output, &mut scale, 8, 1);
            assert!(scale[0] > 0.0);
            assert_eq!(output[0], 127);
            assert_eq!(output[1], -127);
        }

        #[test]
        fn test_quantize_neon_zeros() {
            let cache = [0.0f32; 32];
            let mut output = [0i8; 32];
            let mut scale = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut output, &mut scale, 32, 1);
            assert!(output.iter().all(|&v| v == 0));
        }

        #[test]
        fn test_quantize_neon_multi_position() {
            let cache: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
            let mut output = [0i8; 64];
            let mut scale = [0.0f32; 2];
            kv_cache_quantize_neon(&cache, &mut output, &mut scale, 32, 2);
            assert!(scale[0] > 0.0);
            assert!(scale[1] > 0.0);
        }

        #[test]
        fn test_dequantize_neon_basic() {
            let cache = vec![127i8, -127, 64, -64];
            let scale = vec![1.0 / 127.0];
            let mut output = [0.0f32; 4];
            kv_cache_dequantize_neon(&cache, &scale, &mut output, 4, 1);
            assert!((output[0] - 1.0).abs() < 0.01);
            assert!((output[1] + 1.0).abs() < 0.01);
        }

        // ── Quantize / dequantize round-trip NEON ───────────────────

        #[test]
        fn test_quantize_dequantize_roundtrip_neon_head32() {
            let cache: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
            let mut q_out = [0i8; 32];
            let mut scale = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut q_out, &mut scale, 32, 1);

            let mut deq_out = [0.0f32; 32];
            kv_cache_dequantize_neon(&q_out, &scale, &mut deq_out, 32, 1);

            for (orig, deq) in cache.iter().zip(deq_out.iter()) {
                assert!(
                    (orig - deq).abs() < 0.02,
                    "round-trip error too large: orig={orig} deq={deq}"
                );
            }
        }

        #[test]
        fn test_quantize_dequantize_roundtrip_neon_head64() {
            let cache: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.01).collect();
            let mut q_out = [0i8; 128];
            let mut scale = [0.0f32; 2];
            kv_cache_quantize_neon(&cache, &mut q_out, &mut scale, 64, 2);

            let mut deq_out = [0.0f32; 128];
            kv_cache_dequantize_neon(&q_out, &scale, &mut deq_out, 64, 2);

            for (orig, deq) in cache.iter().zip(deq_out.iter()) {
                assert!(
                    (orig - deq).abs() < 0.02,
                    "round-trip error too large: orig={orig} deq={deq}"
                );
            }
        }

        #[test]
        fn test_quantize_dequantize_roundtrip_neon_head128() {
            let cache: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.003).collect();
            let mut q_out = [0i8; 512];
            let mut scale = [0.0f32; 4];
            kv_cache_quantize_neon(&cache, &mut q_out, &mut scale, 128, 4);

            let mut deq_out = [0.0f32; 512];
            kv_cache_dequantize_neon(&q_out, &scale, &mut deq_out, 128, 4);

            for (orig, deq) in cache.iter().zip(deq_out.iter()) {
                assert!(
                    (orig - deq).abs() < 0.02,
                    "round-trip error too large: orig={orig} deq={deq}"
                );
            }
        }

        // ── Quantize parity (NEON vs scalar) ────────────────────────

        #[test]
        fn test_quantize_parity_head32() {
            let cache: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
            let mut q_neon = [0i8; 32];
            let mut s_neon = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut q_neon, &mut s_neon, 32, 1);

            let mut q_scalar = [0i8; 32];
            let mut s_scalar = [0.0f32; 1];
            kv_cache_quantize_scalar(&cache, &mut q_scalar, &mut s_scalar, 32, 1);

            assert!((s_neon[0] - s_scalar[0]).abs() < 1e-6);
            for (a, b) in q_neon.iter().zip(q_scalar.iter()) {
                assert!((a - b).abs() <= 1, "quantize mismatch: neon={a} scalar={b}");
            }
        }

        #[test]
        fn test_quantize_parity_head64() {
            let cache: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
            let mut q_neon = [0i8; 64];
            let mut s_neon = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut q_neon, &mut s_neon, 64, 1);

            let mut q_scalar = [0i8; 64];
            let mut s_scalar = [0.0f32; 1];
            kv_cache_quantize_scalar(&cache, &mut q_scalar, &mut s_scalar, 64, 1);

            assert!((s_neon[0] - s_scalar[0]).abs() < 1e-6);
            for (a, b) in q_neon.iter().zip(q_scalar.iter()) {
                assert!((a - b).abs() <= 1, "quantize mismatch: neon={a} scalar={b}");
            }
        }

        #[test]
        fn test_quantize_parity_head128() {
            let cache: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.02).collect();
            let mut q_neon = [0i8; 128];
            let mut s_neon = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut q_neon, &mut s_neon, 128, 1);

            let mut q_scalar = [0i8; 128];
            let mut s_scalar = [0.0f32; 1];
            kv_cache_quantize_scalar(&cache, &mut q_scalar, &mut s_scalar, 128, 1);

            assert!((s_neon[0] - s_scalar[0]).abs() < 1e-6);
            for (a, b) in q_neon.iter().zip(q_scalar.iter()) {
                assert!((a - b).abs() <= 1, "quantize mismatch: neon={a} scalar={b}");
            }
        }

        // ── Dequantize parity (NEON vs scalar) ─────────────────────

        #[test]
        fn test_dequantize_parity_head32() {
            let cache: Vec<i8> = (0..32).map(|i| (i as i8).wrapping_sub(16)).collect();
            let scale = vec![0.01f32];
            let mut o_neon = [0.0f32; 32];
            let mut o_scalar = [0.0f32; 32];
            kv_cache_dequantize_neon(&cache, &scale, &mut o_neon, 32, 1);
            kv_cache_dequantize_scalar(&cache, &scale, &mut o_scalar, 32, 1);
            for (a, b) in o_neon.iter().zip(o_scalar.iter()) {
                assert!((a - b).abs() < 1e-6, "dequant mismatch: neon={a} scalar={b}");
            }
        }

        #[test]
        fn test_dequantize_parity_head64() {
            let cache: Vec<i8> = (0..64).map(|i| ((i % 255) as i8).wrapping_sub(64)).collect();
            let scale = vec![0.005f32];
            let mut o_neon = [0.0f32; 64];
            let mut o_scalar = [0.0f32; 64];
            kv_cache_dequantize_neon(&cache, &scale, &mut o_neon, 64, 1);
            kv_cache_dequantize_scalar(&cache, &scale, &mut o_scalar, 64, 1);
            for (a, b) in o_neon.iter().zip(o_scalar.iter()) {
                assert!((a - b).abs() < 1e-6, "dequant mismatch: neon={a} scalar={b}");
            }
        }

        #[test]
        fn test_dequantize_parity_head128() {
            let cache: Vec<i8> = (0..128).map(|i| ((i % 255) as i8).wrapping_sub(64)).collect();
            let scale = vec![0.002f32];
            let mut o_neon = [0.0f32; 128];
            let mut o_scalar = [0.0f32; 128];
            kv_cache_dequantize_neon(&cache, &scale, &mut o_neon, 128, 1);
            kv_cache_dequantize_scalar(&cache, &scale, &mut o_scalar, 128, 1);
            for (a, b) in o_neon.iter().zip(o_scalar.iter()) {
                assert!((a - b).abs() < 1e-6, "dequant mismatch: neon={a} scalar={b}");
            }
        }

        // ── Mixed / integration NEON tests ──────────────────────────

        #[test]
        fn test_append_then_gather_neon() {
            let mut cache = make_cache(8, 32, 1);
            for pos in 0..4 {
                let kv = make_kv_row(32, 1, pos as f32);
                kv_cache_append_neon(&mut cache, &kv, pos, 32, 1);
            }
            let indices = vec![0, 2, 3];
            let mut output = vec![0.0f32; 3 * 32];
            kv_cache_gather_neon(&cache, &indices, &mut output, 32);
            // Verify gathered rows match what was appended.
            for (out_idx, &cache_idx) in indices.iter().enumerate() {
                let expected = make_kv_row(32, 1, cache_idx as f32);
                let start = out_idx * 32;
                assert_eq!(&output[start..start + 32], &expected[..32]);
            }
        }

        #[test]
        fn test_rotate_then_gather_neon() {
            // 4 positions × stride 4 (head_dim=2, num_heads=2)
            let mut cache: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, // pos 0
                5.0, 6.0, 7.0, 8.0, // pos 1
                9.0, 10.0, 11.0, 12.0, // pos 2
                13.0, 14.0, 15.0, 16.0, // pos 3
            ];
            kv_cache_rotate_neon(&mut cache, 1, 2, 2);
            // After rotate by 1: pos1,pos2,pos3,pos0
            let indices = [0]; // should now be old pos 1
            let mut output = [0.0f32; 4];
            // Gather using head_dim * num_heads = 4 as row size
            kv_cache_gather_neon(&cache, &indices, &mut output, 4);
            assert_eq!(output, vec![5.0, 6.0, 7.0, 8.0]);
        }

        #[test]
        fn test_quantize_dequantize_neon_preserves_zero() {
            let cache = [0.0f32; 64];
            let mut q_out = [0i8; 64];
            let mut scale = [0.0f32; 1];
            kv_cache_quantize_neon(&cache, &mut q_out, &mut scale, 64, 1);

            let mut deq_out = [0.0f32; 64];
            kv_cache_dequantize_neon(&q_out, &scale, &mut deq_out, 64, 1);
            assert!(deq_out.iter().all(|&v| v == 0.0));
        }

        #[test]
        fn test_quantize_neon_max_capacity() {
            // Larger cache: 256 positions × 128 head_dim
            let n = 256 * 128;
            let cache: Vec<f32> = (0..n).map(|i| (i as f32 - (n / 2) as f32) * 0.001).collect();
            let mut q_out = vec![0i8; n];
            let mut scale = [0.0f32; 256];
            kv_cache_quantize_neon(&cache, &mut q_out, &mut scale, 128, 256);

            let mut deq_out = vec![0.0f32; n];
            kv_cache_dequantize_neon(&q_out, &scale, &mut deq_out, 128, 256);
            for (orig, deq) in cache.iter().zip(deq_out.iter()) {
                // Tolerance scales with absmax per row (~16.4); int8 step ≈ 0.13.
                assert!(
                    (orig - deq).abs() < 0.15,
                    "round-trip error too large: orig={orig} deq={deq}"
                );
            }
        }
    }
}
