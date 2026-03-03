//! NEON-optimized KV cache operations for attention mechanisms (Apple Silicon / ARM64).
//!
//! Provides append, gather, sliding-window rotation, copy, scale, mask, and
//! concatenation primitives using ARM NEON intrinsics with scalar fallback for
//! remainder elements.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── append ─────────────────────────────────────────────────────────

/// Append new K/V data to `cache` at the given sequence position.
///
/// Cache layout: `[max_seq_len, num_heads, head_dim]` flattened in row-major
/// order.  `new_data` must contain exactly `num_heads * head_dim` elements
/// (one full position).
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_append_f32(
    cache: &mut [f32],
    new_data: &[f32],
    seq_pos: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let stride = num_heads * head_dim;
    assert_eq!(
        new_data.len(),
        stride,
        "new_data length ({}) != num_heads*head_dim ({stride})",
        new_data.len(),
    );
    let offset = seq_pos * stride;
    assert!(
        offset + stride <= cache.len(),
        "cache too small for seq_pos={seq_pos}: need {}, got {}",
        offset + stride,
        cache.len(),
    );

    let dst = &mut cache[offset..offset + stride];
    let chunks = stride / 4;
    let remainder = stride % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(new_data.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), v);
        }
    }

    let tail = chunks * 4;
    dst[tail..tail + remainder].copy_from_slice(&new_data[tail..tail + remainder]);
}

// ── gather ─────────────────────────────────────────────────────────

/// Gather cache entries by index.
///
/// For each index `i` in `indices`, copies `num_heads * head_dim` elements from
/// `cache[indices[i] * stride ..]` into the corresponding region of `output`.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_gather_f32(
    cache: &[f32],
    indices: &[usize],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
) {
    let stride = num_heads * head_dim;
    assert!(
        output.len() >= indices.len() * stride,
        "output too small: need {}, got {}",
        indices.len() * stride,
        output.len(),
    );

    for (out_idx, &cache_idx) in indices.iter().enumerate() {
        let src_off = cache_idx * stride;
        assert!(
            src_off + stride <= cache.len(),
            "gather index {cache_idx} out of bounds (cache holds {} entries)",
            cache.len() / stride,
        );
        let src = &cache[src_off..src_off + stride];
        let dst = &mut output[out_idx * stride..(out_idx + 1) * stride];

        let chunks = stride / 4;
        let remainder = stride % 4;

        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                let v = vld1q_f32(src.as_ptr().add(base));
                vst1q_f32(dst.as_mut_ptr().add(base), v);
            }
        }

        let tail = chunks * 4;
        dst[tail..tail + remainder].copy_from_slice(&src[tail..tail + remainder]);
    }
}

// ── rotate (sliding window) ────────────────────────────────────────

/// Sliding-window rotation: shift `positions` oldest entries out and move the
/// remaining entries to the front.
///
/// `cache` stores `window_size` positions, each of `num_heads * head_dim`
/// elements.  After rotation the first `window_size - positions` entries are the
/// previously newer data, and the trailing `positions` entries are zeroed.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_rotate_f32(
    cache: &mut [f32],
    positions: usize,
    window_size: usize,
    num_heads: usize,
    head_dim: usize,
) {
    let stride = num_heads * head_dim;
    let total = window_size * stride;
    assert!(cache.len() >= total, "cache too small for window: need {total}, got {}", cache.len(),);
    assert!(positions <= window_size, "positions ({positions}) > window_size ({window_size})");

    if positions == 0 || positions == window_size {
        if positions == window_size {
            // Zero the whole window.
            neon_zero(cache, total);
        }
        return;
    }

    let keep = window_size - positions;
    let src_off = positions * stride;
    let copy_len = keep * stride;

    // Move kept entries to front (memmove-safe via copy_within).
    cache.copy_within(src_off..src_off + copy_len, 0);

    // Zero the freed tail with NEON.
    let zero_start = copy_len;
    neon_zero(&mut cache[zero_start..], positions * stride);
}

/// Zero `len` f32 values starting at the beginning of `buf` using NEON.
#[cfg(target_arch = "aarch64")]
#[inline]
fn neon_zero(buf: &mut [f32], len: usize) {
    assert!(buf.len() >= len);
    let chunks = len / 4;
    let remainder = len % 4;
    let zero = unsafe { vdupq_n_f32(0.0) };

    for i in 0..chunks {
        unsafe {
            vst1q_f32(buf.as_mut_ptr().add(i * 4), zero);
        }
    }
    let tail = chunks * 4;
    for i in 0..remainder {
        buf[tail + i] = 0.0;
    }
}

// ── copy ───────────────────────────────────────────────────────────

/// Fast NEON copy of `num_entries` cache positions (each `head_dim` wide).
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_copy_f32(src: &[f32], dst: &mut [f32], num_entries: usize, head_dim: usize) {
    let total = num_entries * head_dim;
    assert!(src.len() >= total, "src too small: need {total}, got {}", src.len(),);
    assert!(dst.len() >= total, "dst too small: need {total}, got {}", dst.len(),);

    let chunks = total / 4;
    let remainder = total % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(src.as_ptr().add(base));
            vst1q_f32(dst.as_mut_ptr().add(base), v);
        }
    }

    let tail = chunks * 4;
    dst[tail..tail + remainder].copy_from_slice(&src[tail..tail + remainder]);
}

// ── scale ──────────────────────────────────────────────────────────

/// Scale `entries` cache positions (each `head_dim` wide) in place.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_scale_f32(cache: &mut [f32], scale: f32, entries: usize, head_dim: usize) {
    let total = entries * head_dim;
    assert!(cache.len() >= total, "cache too small: need {total}, got {}", cache.len(),);

    let data = &mut cache[..total];
    let chunks = total / 4;
    let remainder = total % 4;
    let scale_v = unsafe { vdupq_n_f32(scale) };

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let v = vld1q_f32(data.as_ptr().add(base));
            let scaled = vmulq_f32(v, scale_v);
            vst1q_f32(data.as_mut_ptr().add(base), scaled);
        }
    }

    let tail = chunks * 4;
    for i in 0..remainder {
        data[tail + i] *= scale;
    }
}

// ── mask ───────────────────────────────────────────────────────────

/// Zero out cache entries where `mask[i]` is `true`.
///
/// `mask` has one entry per position; each position spans `head_dim` elements.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_mask_f32(cache: &mut [f32], mask: &[bool], entries: usize, head_dim: usize) {
    assert!(mask.len() >= entries, "mask too short: need {entries}, got {}", mask.len(),);
    let total = entries * head_dim;
    assert!(cache.len() >= total, "cache too small: need {total}, got {}", cache.len(),);

    let zero = unsafe { vdupq_n_f32(0.0) };
    let chunks_per_entry = head_dim / 4;
    let remainder = head_dim % 4;

    for entry in 0..entries {
        if !mask[entry] {
            continue;
        }
        let base = entry * head_dim;

        for c in 0..chunks_per_entry {
            unsafe {
                vst1q_f32(cache.as_mut_ptr().add(base + c * 4), zero);
            }
        }
        let tail = base + chunks_per_entry * 4;
        for i in 0..remainder {
            cache[tail + i] = 0.0;
        }
    }
}

// ── concat ─────────────────────────────────────────────────────────

/// Concatenate multiple caches into `output` in order.
///
/// `caches[i]` contains `entries_per_cache[i] * head_dim` elements.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_concat_f32(
    caches: &[&[f32]],
    output: &mut [f32],
    entries_per_cache: &[usize],
    head_dim: usize,
) {
    assert_eq!(
        caches.len(),
        entries_per_cache.len(),
        "caches.len() ({}) != entries_per_cache.len() ({})",
        caches.len(),
        entries_per_cache.len(),
    );

    let total: usize = entries_per_cache.iter().map(|&e| e * head_dim).sum();
    assert!(output.len() >= total, "output too small: need {total}, got {}", output.len(),);

    let mut write_off = 0;
    for (cache, &num_entries) in caches.iter().zip(entries_per_cache.iter()) {
        let len = num_entries * head_dim;
        assert!(cache.len() >= len, "cache too small: need {len}, got {}", cache.len(),);

        let src = &cache[..len];
        let dst = &mut output[write_off..write_off + len];
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            unsafe {
                let v = vld1q_f32(src.as_ptr().add(base));
                vst1q_f32(dst.as_mut_ptr().add(base), v);
            }
        }

        let tail = chunks * 4;
        dst[tail..tail + remainder].copy_from_slice(&src[tail..tail + remainder]);

        write_off += len;
    }
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────

    fn make_cache(seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
        vec![0.0f32; seq_len * num_heads * head_dim]
    }

    fn sequential(n: usize) -> Vec<f32> {
        (1..=n).map(|i| i as f32).collect()
    }

    // ── append tests ──────────────────────────────────────────────

    #[test]
    fn test_append_pos_zero() {
        let mut cache = make_cache(4, 2, 4);
        let data = sequential(8);
        neon_kv_cache_append_f32(&mut cache, &data, 0, 2, 4);
        assert_eq!(&cache[..8], &data[..]);
    }

    #[test]
    fn test_append_pos_middle() {
        let mut cache = make_cache(4, 2, 4);
        let data = sequential(8);
        neon_kv_cache_append_f32(&mut cache, &data, 2, 2, 4);
        assert_eq!(&cache[16..24], &data[..]);
        assert_eq!(&cache[..16], &vec![0.0f32; 16][..]);
    }

    #[test]
    fn test_append_last_position() {
        let mut cache = make_cache(4, 1, 4);
        let data = vec![9.0, 10.0, 11.0, 12.0];
        neon_kv_cache_append_f32(&mut cache, &data, 3, 1, 4);
        assert_eq!(&cache[12..16], &data[..]);
    }

    #[test]
    fn test_append_single_head_odd_dim() {
        let mut cache = make_cache(2, 1, 5);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        neon_kv_cache_append_f32(&mut cache, &data, 0, 1, 5);
        assert_eq!(&cache[..5], &data[..]);
    }

    #[test]
    fn test_append_multiple_heads() {
        let mut cache = make_cache(3, 4, 8);
        let data = sequential(32);
        neon_kv_cache_append_f32(&mut cache, &data, 1, 4, 8);
        assert_eq!(&cache[32..64], &data[..]);
    }

    #[test]
    fn test_append_overwrites_previous() {
        let mut cache = make_cache(2, 1, 4);
        let d1 = vec![1.0, 2.0, 3.0, 4.0];
        neon_kv_cache_append_f32(&mut cache, &d1, 0, 1, 4);
        let d2 = vec![5.0, 6.0, 7.0, 8.0];
        neon_kv_cache_append_f32(&mut cache, &d2, 0, 1, 4);
        assert_eq!(&cache[..4], &d2[..]);
    }

    #[test]
    fn test_append_sequential_positions() {
        let mut cache = make_cache(4, 1, 4);
        for pos in 0..4 {
            let data: Vec<f32> = (0..4).map(|j| (pos * 4 + j) as f32).collect();
            neon_kv_cache_append_f32(&mut cache, &data, pos, 1, 4);
        }
        let expected: Vec<f32> = (0..16).map(|i| i as f32).collect();
        assert_eq!(cache, expected);
    }

    #[test]
    #[should_panic(expected = "new_data length")]
    fn test_append_wrong_data_len() {
        let mut cache = make_cache(2, 1, 4);
        neon_kv_cache_append_f32(&mut cache, &[1.0, 2.0], 0, 1, 4);
    }

    #[test]
    #[should_panic(expected = "cache too small")]
    fn test_append_out_of_bounds() {
        let mut cache = make_cache(2, 1, 4);
        neon_kv_cache_append_f32(&mut cache, &[1.0; 4], 2, 1, 4);
    }

    #[test]
    fn test_append_head_dim_1() {
        let mut cache = make_cache(4, 2, 1);
        let data = vec![7.0, 8.0];
        neon_kv_cache_append_f32(&mut cache, &data, 1, 2, 1);
        assert_eq!(&cache[2..4], &data[..]);
    }

    #[test]
    fn test_append_large_head_dim() {
        let hd = 128;
        let mut cache = make_cache(2, 1, hd);
        let data: Vec<f32> = (0..hd).map(|i| i as f32).collect();
        neon_kv_cache_append_f32(&mut cache, &data, 0, 1, hd);
        assert_eq!(&cache[..hd], &data[..]);
    }

    // ── gather tests ──────────────────────────────────────────────

    #[test]
    fn test_gather_single_index() {
        let mut cache = make_cache(4, 1, 4);
        let data = vec![10.0, 20.0, 30.0, 40.0];
        neon_kv_cache_append_f32(&mut cache, &data, 2, 1, 4);
        let mut out = vec![0.0f32; 4];
        neon_kv_cache_gather_f32(&cache, &[2], &mut out, 1, 4);
        assert_eq!(out, data);
    }

    #[test]
    fn test_gather_multiple_indices() {
        let mut cache = make_cache(4, 1, 4);
        for pos in 0..4 {
            let d: Vec<f32> = (0..4).map(|j| (pos * 10 + j) as f32).collect();
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 4);
        }
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_gather_f32(&cache, &[3, 1], &mut out, 1, 4);
        assert_eq!(&out[..4], &[30.0, 31.0, 32.0, 33.0]);
        assert_eq!(&out[4..8], &[10.0, 11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_gather_duplicate_indices() {
        let mut cache = make_cache(2, 1, 4);
        let data = vec![5.0, 6.0, 7.0, 8.0];
        neon_kv_cache_append_f32(&mut cache, &data, 0, 1, 4);
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_gather_f32(&cache, &[0, 0], &mut out, 1, 4);
        assert_eq!(&out[..4], &data[..]);
        assert_eq!(&out[4..8], &data[..]);
    }

    #[test]
    fn test_gather_odd_head_dim() {
        let mut cache = make_cache(3, 1, 5);
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        neon_kv_cache_append_f32(&mut cache, &d, 1, 1, 5);
        let mut out = vec![0.0f32; 5];
        neon_kv_cache_gather_f32(&cache, &[1], &mut out, 1, 5);
        assert_eq!(out, d);
    }

    #[test]
    fn test_gather_empty_indices() {
        let cache = make_cache(4, 1, 4);
        let mut out: Vec<f32> = vec![];
        neon_kv_cache_gather_f32(&cache, &[], &mut out, 1, 4);
        assert!(out.is_empty());
    }

    #[test]
    fn test_gather_multi_head() {
        let mut cache = make_cache(3, 2, 4);
        let data = sequential(8); // 2 heads × 4 dim
        neon_kv_cache_append_f32(&mut cache, &data, 0, 2, 4);
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_gather_f32(&cache, &[0], &mut out, 2, 4);
        assert_eq!(out, data);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_gather_out_of_bounds() {
        let cache = make_cache(2, 1, 4);
        let mut out = vec![0.0f32; 4];
        neon_kv_cache_gather_f32(&cache, &[2], &mut out, 1, 4);
    }

    #[test]
    fn test_gather_preserves_order() {
        let mut cache = make_cache(4, 1, 4);
        for pos in 0..4 {
            let d: Vec<f32> = vec![(pos + 1) as f32; 4];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 4);
        }
        let mut out = vec![0.0f32; 16];
        neon_kv_cache_gather_f32(&cache, &[2, 0, 3, 1], &mut out, 1, 4);
        assert_eq!(&out[0..4], &[3.0; 4]);
        assert_eq!(&out[4..8], &[1.0; 4]);
        assert_eq!(&out[8..12], &[4.0; 4]);
        assert_eq!(&out[12..16], &[2.0; 4]);
    }

    // ── rotate tests ──────────────────────────────────────────────

    #[test]
    fn test_rotate_shift_one() {
        let mut cache = vec![
            1.0, 2.0, 3.0, 4.0, // pos 0
            5.0, 6.0, 7.0, 8.0, // pos 1
            9.0, 10.0, 11.0, 12.0, // pos 2
        ];
        neon_kv_cache_rotate_f32(&mut cache, 1, 3, 1, 4);
        assert_eq!(&cache[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&cache[4..8], &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(&cache[8..12], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rotate_shift_two() {
        let mut cache = vec![
            1.0, 2.0, // pos 0
            3.0, 4.0, // pos 1
            5.0, 6.0, // pos 2
            7.0, 8.0, // pos 3
        ];
        neon_kv_cache_rotate_f32(&mut cache, 2, 4, 1, 2);
        assert_eq!(&cache[..2], &[5.0, 6.0]);
        assert_eq!(&cache[2..4], &[7.0, 8.0]);
        assert_eq!(&cache[4..8], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rotate_zero_positions() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let mut cache = original.clone();
        neon_kv_cache_rotate_f32(&mut cache, 0, 2, 1, 2);
        assert_eq!(cache, original);
    }

    #[test]
    fn test_rotate_full_window() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0];
        neon_kv_cache_rotate_f32(&mut cache, 2, 2, 1, 2);
        assert_eq!(cache, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rotate_multi_head() {
        let mut cache = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // pos 0: 2 heads × 4
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, // pos 1
        ];
        neon_kv_cache_rotate_f32(&mut cache, 1, 2, 2, 4);
        assert_eq!(&cache[..8], &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        assert_eq!(&cache[8..16], &[0.0; 8]);
    }

    #[test]
    fn test_rotate_odd_dim() {
        let mut cache = vec![
            1.0, 2.0, 3.0, // pos 0: 1 head × 3
            4.0, 5.0, 6.0, // pos 1
            7.0, 8.0, 9.0, // pos 2
        ];
        neon_kv_cache_rotate_f32(&mut cache, 1, 3, 1, 3);
        assert_eq!(&cache[..3], &[4.0, 5.0, 6.0]);
        assert_eq!(&cache[3..6], &[7.0, 8.0, 9.0]);
        assert_eq!(&cache[6..9], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rotate_wrap_around_incremental() {
        // Simulate filling a window and rotating repeatedly.
        let mut cache = vec![0.0f32; 12]; // window=3, head=1, dim=4
        for pos in 0..3 {
            let d: Vec<f32> = vec![(pos + 1) as f32; 4];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 4);
        }
        // cache: [1,1,1,1, 2,2,2,2, 3,3,3,3]
        neon_kv_cache_rotate_f32(&mut cache, 1, 3, 1, 4);
        // now: [2,2,2,2, 3,3,3,3, 0,0,0,0]
        let d4 = vec![4.0f32; 4];
        neon_kv_cache_append_f32(&mut cache, &d4, 2, 1, 4);
        assert_eq!(&cache[..4], &[2.0; 4]);
        assert_eq!(&cache[4..8], &[3.0; 4]);
        assert_eq!(&cache[8..12], &[4.0; 4]);
    }

    // ── copy tests ────────────────────────────────────────────────

    #[test]
    fn test_copy_basic() {
        let src = sequential(16);
        let mut dst = vec![0.0f32; 16];
        neon_kv_cache_copy_f32(&src, &mut dst, 4, 4);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_partial() {
        let src = sequential(16);
        let mut dst = vec![0.0f32; 8];
        neon_kv_cache_copy_f32(&src, &mut dst, 2, 4);
        assert_eq!(dst, &src[..8]);
    }

    #[test]
    fn test_copy_odd_dim() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut dst = vec![0.0f32; 7];
        neon_kv_cache_copy_f32(&src, &mut dst, 1, 7);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_single_element() {
        let src = vec![42.0];
        let mut dst = vec![0.0f32; 1];
        neon_kv_cache_copy_f32(&src, &mut dst, 1, 1);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_large() {
        let n = 256;
        let src: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; n];
        neon_kv_cache_copy_f32(&src, &mut dst, 1, n);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_zero_entries() {
        let src = vec![1.0, 2.0];
        let mut dst: Vec<f32> = vec![];
        neon_kv_cache_copy_f32(&src, &mut dst, 0, 4);
        assert!(dst.is_empty());
    }

    #[test]
    #[should_panic(expected = "src too small")]
    fn test_copy_src_too_small() {
        let src = vec![1.0, 2.0];
        let mut dst = vec![0.0f32; 4];
        neon_kv_cache_copy_f32(&src, &mut dst, 1, 4);
    }

    #[test]
    fn test_copy_equals_slice_copy() {
        let src = sequential(33); // non-aligned
        let mut dst_neon = vec![0.0f32; 33];
        let mut dst_std = vec![0.0f32; 33];
        neon_kv_cache_copy_f32(&src, &mut dst_neon, 1, 33);
        dst_std.copy_from_slice(&src[..33]);
        assert_eq!(dst_neon, dst_std);
    }

    // ── scale tests ───────────────────────────────────────────────

    #[test]
    fn test_scale_basic() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        neon_kv_cache_scale_f32(&mut cache, 0.5, 2, 4);
        assert_eq!(cache, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    }

    #[test]
    fn test_scale_identity() {
        let original = sequential(12);
        let mut cache = original.clone();
        neon_kv_cache_scale_f32(&mut cache, 1.0, 3, 4);
        assert_eq!(cache, original);
    }

    #[test]
    fn test_scale_zero() {
        let mut cache = sequential(8);
        neon_kv_cache_scale_f32(&mut cache, 0.0, 2, 4);
        assert_eq!(cache, vec![0.0; 8]);
    }

    #[test]
    fn test_scale_negative() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0];
        neon_kv_cache_scale_f32(&mut cache, -1.0, 1, 4);
        assert_eq!(cache, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_scale_odd_dim() {
        let mut cache = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        neon_kv_cache_scale_f32(&mut cache, 0.5, 1, 5);
        assert_eq!(cache, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_scale_partial_entries() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        neon_kv_cache_scale_f32(&mut cache, 2.0, 1, 4);
        assert_eq!(&cache[..4], &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(&cache[4..], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_scale_accuracy() {
        let mut cache = vec![1.0f32 / 3.0; 8];
        let scale = 3.0;
        neon_kv_cache_scale_f32(&mut cache, scale, 2, 4);
        for v in &cache {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scale_large() {
        let n = 512;
        let mut cache: Vec<f32> = vec![2.0; n];
        neon_kv_cache_scale_f32(&mut cache, 0.25, n / 4, 4);
        assert!(cache.iter().all(|&v| (v - 0.5).abs() < 1e-7));
    }

    // ── mask tests ────────────────────────────────────────────────

    #[test]
    fn test_mask_all_true() {
        let mut cache = sequential(8);
        let mask = vec![true, true];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 4);
        assert_eq!(cache, vec![0.0; 8]);
    }

    #[test]
    fn test_mask_all_false() {
        let original = sequential(8);
        let mut cache = original.clone();
        let mask = vec![false, false];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 4);
        assert_eq!(cache, original);
    }

    #[test]
    fn test_mask_selective() {
        let mut cache = vec![
            1.0, 2.0, 3.0, 4.0, // entry 0
            5.0, 6.0, 7.0, 8.0, // entry 1
            9.0, 10.0, 11.0, 12.0, // entry 2
        ];
        let mask = vec![false, true, false];
        neon_kv_cache_mask_f32(&mut cache, &mask, 3, 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cache[4..8], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&cache[8..12], &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_mask_odd_dim() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![true, false];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 3);
        assert_eq!(&cache[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&cache[3..6], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mask_single_entry() {
        let mut cache = vec![42.0, 43.0, 44.0, 45.0];
        let mask = vec![true];
        neon_kv_cache_mask_f32(&mut cache, &mask, 1, 4);
        assert_eq!(cache, vec![0.0; 4]);
    }

    #[test]
    fn test_mask_idempotent() {
        let mut cache = sequential(8);
        let mask = vec![true, true];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 4);
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 4);
        assert_eq!(cache, vec![0.0; 8]);
    }

    #[test]
    fn test_mask_dim_1() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![false, true, false, true];
        neon_kv_cache_mask_f32(&mut cache, &mask, 4, 1);
        assert_eq!(cache, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_mask_alternating() {
        let mut cache = sequential(20);
        let mask: Vec<bool> = (0..5).map(|i| i % 2 == 0).collect();
        neon_kv_cache_mask_f32(&mut cache, &mask, 5, 4);
        assert_eq!(&cache[0..4], &[0.0; 4]); // entry 0 masked
        assert_eq!(&cache[4..8], &[5.0, 6.0, 7.0, 8.0]); // entry 1 kept
        assert_eq!(&cache[8..12], &[0.0; 4]); // entry 2 masked
        assert_eq!(&cache[12..16], &[13.0, 14.0, 15.0, 16.0]); // entry 3 kept
        assert_eq!(&cache[16..20], &[0.0; 4]); // entry 4 masked
    }

    // ── concat tests ──────────────────────────────────────────────

    #[test]
    fn test_concat_two_caches() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_concat_f32(&[&a, &b], &mut out, &[1, 1], 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_three_caches() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let c = vec![5.0, 6.0];
        let mut out = vec![0.0f32; 6];
        neon_kv_cache_concat_f32(&[&a, &b, &c], &mut out, &[1, 1, 1], 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_single_cache() {
        let a = sequential(8);
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_concat_f32(&[&a[..]], &mut out, &[2], 4);
        assert_eq!(out, a);
    }

    #[test]
    fn test_concat_different_entry_counts() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 entries × dim 4
        let b = vec![9.0, 10.0, 11.0, 12.0]; // 1 entry × dim 4
        let mut out = vec![0.0f32; 12];
        neon_kv_cache_concat_f32(&[&a, &b], &mut out, &[2, 1], 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_concat_empty_cache_in_list() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![];
        let c = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_concat_f32(&[&a[..], &b[..], &c[..]], &mut out, &[1, 0, 1], 4);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_preserves_order() {
        let caches: Vec<Vec<f32>> = (0..4).map(|i| vec![(i + 1) as f32; 4]).collect();
        let refs: Vec<&[f32]> = caches.iter().map(|v| v.as_slice()).collect();
        let entries = vec![1usize; 4];
        let mut out = vec![0.0f32; 16];
        neon_kv_cache_concat_f32(&refs, &mut out, &entries, 4);
        for i in 0..4 {
            assert_eq!(&out[i * 4..(i + 1) * 4], &[(i + 1) as f32; 4]);
        }
    }

    #[test]
    fn test_concat_odd_dim() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 1 entry × dim 5
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let mut out = vec![0.0f32; 10];
        neon_kv_cache_concat_f32(&[&a[..], &b[..]], &mut out, &[1, 1], 5);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn test_concat_output_too_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 2];
        neon_kv_cache_concat_f32(&[&a[..]], &mut out, &[1], 4);
    }

    #[test]
    #[should_panic(expected = "caches.len()")]
    fn test_concat_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 4];
        neon_kv_cache_concat_f32(&[&a[..]], &mut out, &[1, 1], 2);
    }

    // ── cross-function integration tests ──────────────────────────

    #[test]
    fn test_append_then_gather_round_trip() {
        let mut cache = make_cache(8, 2, 4);
        for pos in 0..8 {
            let d: Vec<f32> = (0..8).map(|j| (pos * 8 + j) as f32).collect();
            neon_kv_cache_append_f32(&mut cache, &d, pos, 2, 4);
        }
        let mut out = vec![0.0f32; 24]; // gather 3 entries
        neon_kv_cache_gather_f32(&cache, &[7, 0, 3], &mut out, 2, 4);
        // Verify entry 7
        let expected_7: Vec<f32> = (0..8).map(|j| (7 * 8 + j) as f32).collect();
        assert_eq!(&out[..8], &expected_7[..]);
        // Verify entry 0
        let expected_0: Vec<f32> = (0..8).map(|j| j as f32).collect();
        assert_eq!(&out[8..16], &expected_0[..]);
    }

    #[test]
    fn test_copy_then_scale() {
        let src = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let mut dst = vec![0.0f32; 8];
        neon_kv_cache_copy_f32(&src, &mut dst, 2, 4);
        neon_kv_cache_scale_f32(&mut dst, 0.5, 2, 4);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_scale_then_mask() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        neon_kv_cache_scale_f32(&mut cache, 2.0, 2, 4);
        let mask = vec![false, true];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 4);
        assert_eq!(&cache[..4], &[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(&cache[4..8], &[0.0; 4]);
    }

    #[test]
    fn test_rotate_then_append() {
        let mut cache = vec![
            1.0, 2.0, 3.0, 4.0, // pos 0
            5.0, 6.0, 7.0, 8.0, // pos 1
            0.0, 0.0, 0.0, 0.0, // pos 2 (empty)
        ];
        neon_kv_cache_rotate_f32(&mut cache, 1, 3, 1, 4);
        // Now: [5,6,7,8, 0,0,0,0, 0,0,0,0]
        let new_data = vec![9.0, 10.0, 11.0, 12.0];
        neon_kv_cache_append_f32(&mut cache, &new_data, 1, 1, 4);
        assert_eq!(&cache[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&cache[4..8], &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_concat_then_copy() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut combined = vec![0.0f32; 8];
        neon_kv_cache_concat_f32(&[&a, &b], &mut combined, &[1, 1], 4);
        let mut copy = vec![0.0f32; 8];
        neon_kv_cache_copy_f32(&combined, &mut copy, 2, 4);
        assert_eq!(copy, combined);
    }

    #[test]
    fn test_mask_then_gather() {
        let mut cache = make_cache(3, 1, 4);
        for pos in 0..3 {
            let d = vec![(pos + 1) as f32; 4];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 4);
        }
        let mask = vec![false, true, false];
        neon_kv_cache_mask_f32(&mut cache, &mask, 3, 4);
        let mut out = vec![0.0f32; 4];
        neon_kv_cache_gather_f32(&cache, &[1], &mut out, 1, 4);
        assert_eq!(out, vec![0.0; 4]); // masked entry is zero
    }

    #[test]
    fn test_sliding_window_pipeline() {
        // Simulate a full sliding-window scenario: fill, rotate, refill.
        let window = 4;
        let hd = 4;
        let mut cache = make_cache(window, 1, hd);

        // Fill all 4 positions.
        for pos in 0..window {
            let d = vec![(pos + 1) as f32; hd];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, hd);
        }

        // Rotate out the oldest 2 entries.
        neon_kv_cache_rotate_f32(&mut cache, 2, window, 1, hd);

        // Append new entries into the freed slots.
        let d5 = vec![5.0f32; hd];
        let d6 = vec![6.0f32; hd];
        neon_kv_cache_append_f32(&mut cache, &d5, 2, 1, hd);
        neon_kv_cache_append_f32(&mut cache, &d6, 3, 1, hd);

        // Verify final state: [3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6]
        assert_eq!(&cache[0..4], &[3.0; 4]);
        assert_eq!(&cache[4..8], &[4.0; 4]);
        assert_eq!(&cache[8..12], &[5.0; 4]);
        assert_eq!(&cache[12..16], &[6.0; 4]);
    }

    #[test]
    fn test_gather_then_concat() {
        let mut cache = make_cache(4, 1, 4);
        for pos in 0..4 {
            let d = vec![(pos + 1) as f32; 4];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 4);
        }
        let mut gathered_a = vec![0.0f32; 8];
        neon_kv_cache_gather_f32(&cache, &[0, 2], &mut gathered_a, 1, 4);
        let mut gathered_b = vec![0.0f32; 4];
        neon_kv_cache_gather_f32(&cache, &[3], &mut gathered_b, 1, 4);

        let mut out = vec![0.0f32; 12];
        neon_kv_cache_concat_f32(&[&gathered_a, &gathered_b], &mut out, &[2, 1], 4);
        assert_eq!(&out[0..4], &[1.0; 4]);
        assert_eq!(&out[4..8], &[3.0; 4]);
        assert_eq!(&out[8..12], &[4.0; 4]);
    }

    // ── additional coverage tests ─────────────────────────────────

    #[test]
    fn test_append_dim_16_aligned() {
        let mut cache = make_cache(2, 1, 16);
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        neon_kv_cache_append_f32(&mut cache, &data, 0, 1, 16);
        assert_eq!(&cache[..16], &data[..]);
    }

    #[test]
    fn test_append_dim_3_non_aligned() {
        let mut cache = make_cache(2, 1, 3);
        let data = vec![10.0, 20.0, 30.0];
        neon_kv_cache_append_f32(&mut cache, &data, 1, 1, 3);
        assert_eq!(&cache[3..6], &data[..]);
    }

    #[test]
    fn test_gather_all_positions() {
        let mut cache = make_cache(4, 1, 2);
        for pos in 0..4 {
            let d = vec![pos as f32; 2];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 2);
        }
        let mut out = vec![0.0f32; 8];
        neon_kv_cache_gather_f32(&cache, &[0, 1, 2, 3], &mut out, 1, 2);
        for pos in 0..4 {
            assert_eq!(&out[pos * 2..(pos + 1) * 2], &[pos as f32; 2]);
        }
    }

    #[test]
    fn test_gather_reverse_order() {
        let mut cache = make_cache(3, 1, 4);
        for pos in 0..3 {
            let d = vec![(pos + 1) as f32; 4];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, 4);
        }
        let mut out = vec![0.0f32; 12];
        neon_kv_cache_gather_f32(&cache, &[2, 1, 0], &mut out, 1, 4);
        assert_eq!(&out[0..4], &[3.0; 4]);
        assert_eq!(&out[4..8], &[2.0; 4]);
        assert_eq!(&out[8..12], &[1.0; 4]);
    }

    #[test]
    fn test_rotate_single_window() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0];
        neon_kv_cache_rotate_f32(&mut cache, 1, 1, 1, 4);
        assert_eq!(cache, vec![0.0; 4]);
    }

    #[test]
    fn test_copy_two_entries_dim_8() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = vec![0.0f32; 16];
        neon_kv_cache_copy_f32(&src, &mut dst, 2, 8);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_scale_very_small() {
        let mut cache = vec![1e10, 2e10, 3e10, 4e10];
        neon_kv_cache_scale_f32(&mut cache, 1e-10, 1, 4);
        for (i, &v) in cache.iter().enumerate() {
            let expected = (i + 1) as f32;
            assert!((v - expected).abs() < 0.01, "index {i}: {v} != {expected}");
        }
    }

    #[test]
    fn test_mask_large_head_dim() {
        let hd = 64;
        let mut cache: Vec<f32> = (0..(2 * hd) as u32).map(|i| i as f32).collect();
        let mask = vec![true, false];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, hd);
        assert!(cache[..hd].iter().all(|&v| v == 0.0));
        for i in 0..hd {
            assert_eq!(cache[hd + i], (hd + i) as f32);
        }
    }

    #[test]
    fn test_concat_many_small_caches() {
        let caches: Vec<Vec<f32>> = (0..8).map(|i| vec![(i + 1) as f32; 4]).collect();
        let refs: Vec<&[f32]> = caches.iter().map(|v| v.as_slice()).collect();
        let entries = vec![1usize; 8];
        let mut out = vec![0.0f32; 32];
        neon_kv_cache_concat_f32(&refs, &mut out, &entries, 4);
        for i in 0..8 {
            assert_eq!(&out[i * 4..(i + 1) * 4], &[(i + 1) as f32; 4]);
        }
    }

    #[test]
    fn test_append_and_scale_roundtrip() {
        let mut cache = make_cache(2, 1, 4);
        let data = vec![2.0, 4.0, 6.0, 8.0];
        neon_kv_cache_append_f32(&mut cache, &data, 0, 1, 4);
        neon_kv_cache_scale_f32(&mut cache, 0.5, 1, 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mask_then_scale_is_still_zero() {
        let mut cache = sequential(8);
        let mask = vec![true, true];
        neon_kv_cache_mask_f32(&mut cache, &mask, 2, 4);
        neon_kv_cache_scale_f32(&mut cache, 100.0, 2, 4);
        assert_eq!(cache, vec![0.0; 8]);
    }

    #[test]
    fn test_rotate_preserves_unaffected_tail() {
        // Cache larger than window — data beyond window must be untouched.
        let mut cache = vec![
            1.0, 2.0, // pos 0 (window)
            3.0, 4.0, // pos 1 (window)
            99.0, 88.0, // beyond window
        ];
        neon_kv_cache_rotate_f32(&mut cache, 1, 2, 1, 2);
        assert_eq!(&cache[0..2], &[3.0, 4.0]);
        assert_eq!(&cache[2..4], &[0.0, 0.0]);
        assert_eq!(&cache[4..6], &[99.0, 88.0]); // untouched
    }

    #[test]
    fn test_copy_dim_2() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 6];
        neon_kv_cache_copy_f32(&src, &mut dst, 3, 2);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_gather_large_dim() {
        let hd = 128;
        let mut cache = make_cache(2, 1, hd);
        let d: Vec<f32> = (0..hd).map(|i| i as f32).collect();
        neon_kv_cache_append_f32(&mut cache, &d, 0, 1, hd);
        let mut out = vec![0.0f32; hd];
        neon_kv_cache_gather_f32(&cache, &[0], &mut out, 1, hd);
        assert_eq!(out, d);
    }

    #[test]
    fn test_scale_double_then_halve() {
        let original = sequential(8);
        let mut cache = original.clone();
        neon_kv_cache_scale_f32(&mut cache, 2.0, 2, 4);
        neon_kv_cache_scale_f32(&mut cache, 0.5, 2, 4);
        assert_eq!(cache, original);
    }

    #[test]
    fn test_concat_all_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        neon_kv_cache_concat_f32(&[&a[..], &b[..]], &mut out, &[0, 0], 4);
        assert!(out.is_empty());
    }

    #[test]
    fn test_append_all_positions_then_mask_even() {
        let n = 8;
        let hd = 4;
        let mut cache = make_cache(n, 1, hd);
        for pos in 0..n {
            let d = vec![(pos + 1) as f32; hd];
            neon_kv_cache_append_f32(&mut cache, &d, pos, 1, hd);
        }
        let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        neon_kv_cache_mask_f32(&mut cache, &mask, n, hd);
        for pos in 0..n {
            let slice = &cache[pos * hd..(pos + 1) * hd];
            if pos % 2 == 0 {
                assert_eq!(slice, &[0.0; 4]);
            } else {
                assert_eq!(slice, &[(pos + 1) as f32; 4]);
            }
        }
    }

    #[test]
    fn test_copy_preserves_beyond_num_entries() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![99.0f32; 8];
        neon_kv_cache_copy_f32(&src, &mut dst, 1, 4);
        assert_eq!(&dst[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&dst[4..], &[99.0, 99.0, 99.0, 99.0]); // untouched
    }
}
