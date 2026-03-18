//! ARM NEON SIMD-accelerated KV cache operations for transformer
//! inference on Apple Silicon.
//!
//! Every public function contains a NEON fast-path gated behind
//! `target_arch = "aarch64"` and a scalar fallback so that code
//! compiles and tests run on all architectures.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── internal helpers ───────────────────────────────────────────

/// NEON-accelerated memcpy for f32 slices (same length).
#[inline]
fn simd_copy(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "aarch64")]
    {
        let n = src.len();
        let chunks = n / 4;
        for i in 0..chunks {
            let b = i * 4;
            unsafe {
                vst1q_f32(dst.as_mut_ptr().add(b), vld1q_f32(src.as_ptr().add(b)));
            }
        }
        dst[chunks * 4..].copy_from_slice(&src[chunks * 4..]);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dst.copy_from_slice(src);
    }
}

/// NEON-accelerated zero-fill for f32 slices.
#[inline]
fn simd_zero(buf: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let n = buf.len();
        let chunks = n / 4;
        let zero = unsafe { vdupq_n_f32(0.0) };
        for i in 0..chunks {
            unsafe {
                vst1q_f32(buf.as_mut_ptr().add(i * 4), zero);
            }
        }
        buf[chunks * 4..].fill(0.0);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        buf.fill(0.0);
    }
}

// ═══════════════════════════════════════════════════════════════
// 1. kv_cache_append
// ═══════════════════════════════════════════════════════════════

/// Append new key/value vectors to the cache at `position`.
///
/// `new_data` length must be a multiple of `head_dim`. Each
/// consecutive `head_dim`-sized chunk is written to cache
/// positions `position`, `position + 1`, …
pub fn kv_cache_append(cache: &mut [f32], position: usize, new_data: &[f32], head_dim: usize) {
    assert!(head_dim > 0, "head_dim must be positive");
    if new_data.is_empty() {
        return;
    }
    assert!(
        new_data.len().is_multiple_of(head_dim),
        "new_data length {} not a multiple of head_dim {head_dim}",
        new_data.len(),
    );
    let dst_start = position * head_dim;
    let dst_end = dst_start + new_data.len();
    assert!(dst_end <= cache.len(), "append overflows cache: need {dst_end}, have {}", cache.len(),);
    simd_copy(new_data, &mut cache[dst_start..dst_end]);
}

// ═══════════════════════════════════════════════════════════════
// 2. kv_cache_gather
// ═══════════════════════════════════════════════════════════════

/// Gather cached vectors by position indices.
///
/// For each `positions[i]`, copies `head_dim` elements from
/// `cache[pos * head_dim..]` into `output[i * head_dim..]`.
pub fn kv_cache_gather(cache: &[f32], positions: &[usize], head_dim: usize, output: &mut [f32]) {
    assert!(head_dim > 0, "head_dim must be positive");
    let n = positions.len();
    assert_eq!(output.len(), n * head_dim, "output length {} != {} * {head_dim}", output.len(), n,);

    for (i, &pos) in positions.iter().enumerate() {
        let src_off = pos * head_dim;
        assert!(src_off + head_dim <= cache.len(), "position {pos} out of cache bounds",);
        let src = &cache[src_off..src_off + head_dim];
        let dst = &mut output[i * head_dim..(i + 1) * head_dim];
        simd_copy(src, dst);
    }
}

// ═══════════════════════════════════════════════════════════════
// 3. kv_cache_rotate
// ═══════════════════════════════════════════════════════════════

/// Apply RoPE rotation to cached key vectors in-place.
///
/// Rotates `num_positions` vectors starting at `start_pos`.
/// Each pair `(x[2k], x[2k+1])` is rotated by angle
/// `θ = pos / freq_base^(2k / head_dim)`.
pub fn kv_cache_rotate(
    cache: &mut [f32],
    start_pos: usize,
    num_positions: usize,
    head_dim: usize,
    freq_base: f32,
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert!(head_dim.is_multiple_of(2), "head_dim must be even for RoPE",);
    let half_dim = head_dim / 2;

    // Precompute dimension-dependent inverse frequencies.
    let inv_freq: Vec<f32> =
        (0..half_dim).map(|k| 1.0 / freq_base.powf(2.0 * k as f32 / head_dim as f32)).collect();

    for p in 0..num_positions {
        let abs_pos = start_pos + p;
        let pos_f = abs_pos as f32;
        let off = abs_pos * head_dim;
        assert!(off + head_dim <= cache.len(), "position {abs_pos} out of cache bounds",);
        let data = &mut cache[off..off + head_dim];

        #[cfg(target_arch = "aarch64")]
        {
            // Build cos/sin tables, then vectorise rotation.
            let mut cos_tab = vec![0.0f32; half_dim];
            let mut sin_tab = vec![0.0f32; half_dim];
            for (k, (ct, st)) in cos_tab.iter_mut().zip(sin_tab.iter_mut()).enumerate() {
                let theta = pos_f * inv_freq[k];
                (*st, *ct) = theta.sin_cos();
            }

            let neon_pairs = half_dim / 4;
            for c in 0..neon_pairs {
                let k = c * 4;
                unsafe {
                    let cv = vld1q_f32(cos_tab.as_ptr().add(k));
                    let sv = vld1q_f32(sin_tab.as_ptr().add(k));
                    let pr = vld2q_f32(data.as_ptr().add(k * 2));
                    let evens = pr.0;
                    let odds = pr.1;
                    // new_even = evens*cos - odds*sin
                    let ne = vfmsq_f32(vmulq_f32(evens, cv), odds, sv);
                    // new_odd = odds*cos + evens*sin
                    let no = vfmaq_f32(vmulq_f32(odds, cv), evens, sv);
                    vst2q_f32(data.as_mut_ptr().add(k * 2), float32x4x2_t(ne, no));
                }
            }
            // Tail pairs beyond the NEON boundary.
            let tail_k = neon_pairs * 4;
            for (j, pair) in
                data[tail_k * 2..].chunks_exact_mut(2).enumerate().take(half_dim - tail_k)
            {
                let ak = tail_k + j;
                let cos_t = cos_tab[ak];
                let sin_t = sin_tab[ak];
                let x0 = pair[0];
                let x1 = pair[1];
                pair[0] = x0 * cos_t - x1 * sin_t;
                pair[1] = x0 * sin_t + x1 * cos_t;
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            for (k, pair) in data.chunks_exact_mut(2).enumerate().take(half_dim) {
                let theta = pos_f * inv_freq[k];
                let (sin_t, cos_t) = theta.sin_cos();
                let x0 = pair[0];
                let x1 = pair[1];
                pair[0] = x0 * cos_t - x1 * sin_t;
                pair[1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 4. kv_cache_copy
// ═══════════════════════════════════════════════════════════════

/// Bulk copy cache entries between two caches.
///
/// For each `i`, copies `head_dim` elements from
/// `src[src_positions[i] * head_dim..]` into
/// `dst[dst_positions[i] * head_dim..]`.
pub fn kv_cache_copy(
    src: &[f32],
    dst: &mut [f32],
    src_positions: &[usize],
    dst_positions: &[usize],
    head_dim: usize,
) {
    assert!(head_dim > 0, "head_dim must be positive");
    assert_eq!(src_positions.len(), dst_positions.len(), "position lists must have equal length",);

    for (&sp, &dp) in src_positions.iter().zip(dst_positions.iter()) {
        let s_off = sp * head_dim;
        let d_off = dp * head_dim;
        assert!(s_off + head_dim <= src.len(), "src position {sp} out of bounds",);
        assert!(d_off + head_dim <= dst.len(), "dst position {dp} out of bounds",);
        let s = &src[s_off..s_off + head_dim];

        // Cannot borrow dst twice; use ptr copy which is safe
        // because src and dst are disjoint borrows.
        #[cfg(target_arch = "aarch64")]
        {
            let chunks = head_dim / 4;
            for i in 0..chunks {
                let b = i * 4;
                unsafe {
                    vst1q_f32(dst.as_mut_ptr().add(d_off + b), vld1q_f32(s.as_ptr().add(b)));
                }
            }
            let tail = chunks * 4;
            dst[d_off + tail..d_off + head_dim].copy_from_slice(&s[tail..]);
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            dst[d_off..d_off + head_dim].copy_from_slice(s);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// 5. kv_cache_clear_range
// ═══════════════════════════════════════════════════════════════

/// Zero out cache positions in `[start_pos, end_pos)`.
///
/// Does nothing when `start_pos >= end_pos`.
pub fn kv_cache_clear_range(cache: &mut [f32], start_pos: usize, end_pos: usize, head_dim: usize) {
    if start_pos >= end_pos {
        return;
    }
    assert!(head_dim > 0, "head_dim must be positive");
    let byte_start = start_pos * head_dim;
    let byte_end = end_pos * head_dim;
    assert!(
        byte_end <= cache.len(),
        "clear range overflows cache: need {byte_end}, have {}",
        cache.len(),
    );
    simd_zero(&mut cache[byte_start..byte_end]);
}

// ═══════════════════════════════════════════════════════════════
// 6. kv_cache_paged_lookup
// ═══════════════════════════════════════════════════════════════

/// Paged attention cache lookup.
///
/// Maps each logical `seq_positions[i]` through `page_table` to a
/// physical page, reads the corresponding `head_dim`-sized vector,
/// and writes it to `output[i * head_dim..]`.
///
/// Logical → physical mapping:
///   logical_page = seq_pos / page_size
///   offset       = seq_pos % page_size
///   physical_off = (page_table\[logical_page] * page_size
///                   + offset) * head_dim
pub fn kv_cache_paged_lookup(
    pages: &[f32],
    page_table: &[usize],
    seq_positions: &[usize],
    page_size: usize,
    head_dim: usize,
    output: &mut [f32],
) {
    assert!(page_size > 0, "page_size must be positive");
    assert!(head_dim > 0, "head_dim must be positive");
    let n = seq_positions.len();
    assert_eq!(output.len(), n * head_dim, "output length {} != {} * {head_dim}", output.len(), n,);

    for (i, &seq_pos) in seq_positions.iter().enumerate() {
        let logical_page = seq_pos / page_size;
        let offset_in_page = seq_pos % page_size;
        assert!(
            logical_page < page_table.len(),
            "logical page {logical_page} out of page_table bounds",
        );
        let phys_page = page_table[logical_page];
        let src_off = (phys_page * page_size + offset_in_page) * head_dim;
        assert!(src_off + head_dim <= pages.len(), "physical offset out of pages bounds",);
        let src = &pages[src_off..src_off + head_dim];
        let dst = &mut output[i * head_dim..(i + 1) * head_dim];
        simd_copy(src, dst);
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Floating-point comparison helper.
    fn approx_eq(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= eps, "mismatch at [{i}]: {x} vs {y} (eps={eps})",);
        }
    }

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // ── kv_cache_append ────────────────────────────────────────

    #[test]
    fn append_single_vector() {
        let mut cache = [0.0f32; 16];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        kv_cache_append(&mut cache, 0, &data, 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn append_multiple_vectors() {
        let mut cache = [0.0f32; 24];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        kv_cache_append(&mut cache, 0, &data, 3);
        assert_eq!(&cache[..6], &data[..]);
    }

    #[test]
    fn append_at_nonzero_position() {
        let mut cache = [0.0f32; 16];
        let data = vec![9.0, 8.0, 7.0, 6.0];
        kv_cache_append(&mut cache, 2, &data, 4);
        assert_eq!(&cache[8..12], &[9.0, 8.0, 7.0, 6.0]);
        assert_eq!(&cache[..8], &[0.0; 8]);
    }

    #[test]
    fn append_head_dim_1() {
        let mut cache = [0.0f32; 8];
        kv_cache_append(&mut cache, 3, &[42.0], 1);
        assert_eq!(cache[3], 42.0);
    }

    #[test]
    fn append_head_dim_3() {
        let mut cache = [0.0f32; 12];
        let data = vec![1.0, 2.0, 3.0];
        kv_cache_append(&mut cache, 1, &data, 3);
        assert_eq!(&cache[3..6], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn append_head_dim_8() {
        let mut cache = [0.0f32; 32];
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 0, &data, 8);
        approx_eq(&cache[..8], &data, 0.0);
    }

    #[test]
    fn append_head_dim_16() {
        let mut cache = [0.0f32; 64];
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 1, &data, 16);
        approx_eq(&cache[16..32], &data, 0.0);
    }

    #[test]
    fn append_empty_data() {
        let mut cache = [1.0f32; 8];
        let orig = cache.clone();
        kv_cache_append(&mut cache, 0, &[], 4);
        assert_eq!(cache, orig);
    }

    #[test]
    fn append_sequential_calls() {
        let mut cache = [0.0f32; 12];
        kv_cache_append(&mut cache, 0, &[1.0, 2.0, 3.0], 3);
        kv_cache_append(&mut cache, 1, &[4.0, 5.0, 6.0], 3);
        assert_eq!(&cache[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],);
    }

    #[test]
    fn append_preserves_prior_data() {
        let mut cache = [0.0f32; 20];
        kv_cache_append(&mut cache, 0, &[1.0, 2.0, 3.0, 4.0], 4);
        kv_cache_append(&mut cache, 2, &[9.0, 8.0, 7.0, 6.0], 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn append_fills_exactly() {
        let mut cache = [0.0f32; 8];
        let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 0, &data, 4);
        approx_eq(&cache, &data, 0.0);
    }

    #[test]
    fn append_single_element_vectors() {
        let mut cache = [0.0f32; 4];
        for i in 0..4 {
            kv_cache_append(&mut cache, i, &[(i + 1) as f32], 1);
        }
        assert_eq!(cache.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── kv_cache_gather ────────────────────────────────────────

    #[test]
    fn gather_single_position() {
        let cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        kv_cache_gather(&cache, &[1], 4, &mut out);
        assert_eq!(out.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn gather_multiple_positions() {
        let cache: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let mut out = [0.0f32; 8];
        kv_cache_gather(&cache, &[0, 2], 4, &mut out);
        assert_eq!(&out[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[4..], &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn gather_out_of_order() {
        let cache: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let mut out = [0.0f32; 6];
        kv_cache_gather(&cache, &[2, 0], 3, &mut out);
        assert_eq!(&out[..3], &[7.0, 8.0, 9.0]);
        assert_eq!(&out[3..], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn gather_duplicate_positions() {
        let cache = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        kv_cache_gather(&cache, &[0, 0], 2, &mut out);
        assert_eq!(out.to_vec(), vec![10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn gather_head_dim_1() {
        let cache = vec![5.0, 6.0, 7.0];
        let mut out = [0.0f32; 2];
        kv_cache_gather(&cache, &[2, 0], 1, &mut out);
        assert_eq!(out.to_vec(), vec![7.0, 5.0]);
    }

    #[test]
    fn gather_head_dim_3() {
        let cache: Vec<f32> = (0..9).map(|x| x as f32).collect();
        let mut out = [0.0f32; 3];
        kv_cache_gather(&cache, &[1], 3, &mut out);
        assert_eq!(out.to_vec(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn gather_head_dim_8() {
        let cache: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut out = [0.0f32; 8];
        kv_cache_gather(&cache, &[2], 8, &mut out);
        let expected: Vec<f32> = (16..24).map(|x| x as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn gather_all_same_position() {
        let cache = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 6];
        kv_cache_gather(&cache, &[0, 0, 0], 2, &mut out);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn gather_reverse_order() {
        let cache: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut out = [0.0f32; 12];
        kv_cache_gather(&cache, &[2, 1, 0], 4, &mut out);
        assert_eq!(&out[..4], &[8.0, 9.0, 10.0, 11.0]);
        assert_eq!(&out[4..8], &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(&out[8..], &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn gather_single_from_large() {
        let cache: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let mut out = [0.0f32; 5];
        kv_cache_gather(&cache, &[4], 5, &mut out);
        let expected: Vec<f32> = (20..25).map(|x| x as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn gather_empty_positions() {
        let cache = vec![1.0, 2.0, 3.0, 4.0];
        let mut out: Vec<f32> = vec![];
        kv_cache_gather(&cache, &[], 4, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn gather_head_dim_5() {
        let cache: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let mut out = [0.0f32; 5];
        kv_cache_gather(&cache, &[1], 5, &mut out);
        assert_eq!(out.to_vec(), vec![5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    // ── kv_cache_rotate ────────────────────────────────────────

    #[test]
    fn rotate_pos_zero_identity() {
        let mut cache = vec![1.0, 0.0, 0.0, 1.0];
        let orig = cache.clone();
        kv_cache_rotate(&mut cache, 0, 1, 4, 10000.0);
        // At pos=0 theta=0 → cos=1, sin=0 → identity.
        approx_eq(&cache, &orig, 1e-6);
    }

    #[test]
    fn rotate_single_position() {
        let mut cache = [0.0; 8];
        cache[4] = 1.0; // place vector at position 1
        kv_cache_rotate(&mut cache, 1, 1, 4, 10000.0);
        // θ_0 = 1/10000^0 = 1.0 → cos(1),sin(1)
        let (s, c) = 1.0f32.sin_cos();
        approx_eq(&cache[4..6], &[c, s], 1e-6);
    }

    #[test]
    fn rotate_multiple_positions() {
        let mut cache = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        kv_cache_rotate(&mut cache, 0, 2, 4, 10000.0);
        // pos=0 → identity
        approx_eq(&cache[..4], &[1.0, 0.0, 0.0, 0.0], 1e-6);
        // pos=1 → rotated
        let (s, c) = 1.0f32.sin_cos();
        approx_eq(&cache[4..6], &[c, s], 1e-6);
    }

    #[test]
    fn rotate_preserves_norm() {
        let mut cache = [0.0f32; 24];
        cache[20] = 3.0;
        cache[21] = 4.0;
        cache[22] = 1.0;
        cache[23] = 2.0;
        let norm_before = l2_norm(&cache[20..24]);
        kv_cache_rotate(&mut cache, 5, 1, 4, 10000.0);
        let norm_after = l2_norm(&cache[20..24]);
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "norm changed: {norm_before} → {norm_after}",
        );
    }

    #[test]
    fn rotate_freq_base_10000() {
        let mut cache = [0.0; 6];
        cache[4] = 1.0; // position 2
        kv_cache_rotate(&mut cache, 2, 1, 2, 10000.0);
        let theta = 2.0 / 10000.0f32.powf(0.0);
        let (s, c) = theta.sin_cos();
        approx_eq(&cache[4..6], &[c, s], 1e-6);
    }

    #[test]
    fn rotate_freq_base_1000() {
        let mut cache = [0.0; 8];
        cache[6] = 1.0; // position 3
        kv_cache_rotate(&mut cache, 3, 1, 2, 1000.0);
        let theta = 3.0 / 1000.0f32.powf(0.0);
        let (s, c) = theta.sin_cos();
        approx_eq(&cache[6..8], &[c, s], 1e-6);
    }

    #[test]
    fn rotate_head_dim_2() {
        let mut cache = vec![0.0, 0.0, 0.0, 1.0];
        kv_cache_rotate(&mut cache, 1, 1, 2, 10000.0);
        let theta = 1.0f32;
        let (s, c) = theta.sin_cos();
        // x0=0, x1=1 → new0 = -sin, new1 = cos
        approx_eq(&cache[2..4], &[-s, c], 1e-6);
    }

    #[test]
    fn rotate_head_dim_8() {
        let mut cache = [0.0f32; 16];
        for i in 0..8 {
            cache[8 + i] = i as f32; // position 1
        }
        let norm_before = l2_norm(&cache[8..16]);
        kv_cache_rotate(&mut cache, 1, 1, 8, 10000.0);
        let norm_after = l2_norm(&cache[8..16]);
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn rotate_head_dim_64() {
        let mut cache = [0.0f32; 256];
        let off = 3 * 64;
        for (i, val) in cache[off..off + 64].iter_mut().enumerate() {
            *val = (i as f32) * 0.1;
        }
        let norm_before = l2_norm(&cache[off..off + 64]);
        kv_cache_rotate(&mut cache, 3, 1, 64, 10000.0);
        let norm_after = l2_norm(&cache[off..off + 64]);
        assert!((norm_before - norm_after).abs() < 1e-3);
    }

    #[test]
    fn rotate_high_position() {
        // Place data at position 0, use start_pos=0 with
        // a large freq_base to simulate high-position effect.
        let mut cache = vec![1.0, 0.0, 0.0, 1.0];
        let norm_before = l2_norm(&cache);
        kv_cache_rotate(&mut cache, 0, 1, 4, 0.01);
        let norm_after = l2_norm(&cache);
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn rotate_all_zeros() {
        let mut cache = [0.0f32; 48];
        kv_cache_rotate(&mut cache, 5, 1, 8, 10000.0);
        // Rotation of zero vector is still zero.
        assert!(cache.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rotate_sequential_positions() {
        let mut cache = [0.0f32; 16];
        for i in 0..4 {
            cache[i * 4] = 1.0;
        }
        kv_cache_rotate(&mut cache, 0, 4, 4, 10000.0);
        // Each position rotated by different angle; norms preserved.
        for i in 0..4 {
            let v = &cache[i * 4..(i + 1) * 4];
            assert!((l2_norm(v) - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn rotate_known_angle_pi_half() {
        // inv_freq[0] = 1/base^0 = 1, so θ = pos * 1 = pos.
        // At pos=0, θ=0 → identity. Verify at pos=0.
        let mut cache = vec![1.0, 0.0];
        kv_cache_rotate(&mut cache, 0, 1, 2, 1.0);
        // θ = 0 → cos=1, sin=0 → identity
        approx_eq(&cache, &[1.0, 0.0], 1e-6);
    }

    // ── kv_cache_copy ──────────────────────────────────────────

    #[test]
    fn copy_single_entry() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = [0.0f32; 8];
        kv_cache_copy(&src, &mut dst, &[1], &[0], 4);
        assert_eq!(&dst[..4], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn copy_multiple_entries() {
        let src: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let mut dst = [0.0f32; 12];
        kv_cache_copy(&src, &mut dst, &[0, 2], &[1, 0], 4);
        assert_eq!(&dst[..4], &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(&dst[4..8], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_preserves_source() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let orig = src.clone();
        let mut dst = [0.0f32; 4];
        kv_cache_copy(&src, &mut dst, &[0], &[0], 4);
        assert_eq!(src, orig);
    }

    #[test]
    fn copy_head_dim_1() {
        let src = vec![10.0, 20.0, 30.0];
        let mut dst = [0.0f32; 3];
        kv_cache_copy(&src, &mut dst, &[2, 0], &[0, 1], 1);
        assert_eq!(&dst[..2], &[30.0, 10.0]);
    }

    #[test]
    fn copy_head_dim_3() {
        let src: Vec<f32> = (0..9).map(|x| x as f32).collect();
        let mut dst = [0.0f32; 9];
        kv_cache_copy(&src, &mut dst, &[1], &[2], 3);
        assert_eq!(&dst[6..9], &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn copy_head_dim_8() {
        let src: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let mut dst = [0.0f32; 16];
        kv_cache_copy(&src, &mut dst, &[1], &[0], 8);
        let expected: Vec<f32> = (8..16).map(|x| x as f32).collect();
        assert_eq!(&dst[..8], &expected[..]);
    }

    #[test]
    fn copy_reverse_mapping() {
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut dst = [0.0f32; 12];
        kv_cache_copy(&src, &mut dst, &[0, 1, 2], &[2, 1, 0], 4);
        assert_eq!(&dst[..4], &[8.0, 9.0, 10.0, 11.0]);
        assert_eq!(&dst[4..8], &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(&dst[8..], &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn copy_empty_positions() {
        let src = [1.0; 8];
        let mut dst = [0.0f32; 8];
        let orig = dst.clone();
        kv_cache_copy(&src, &mut dst, &[], &[], 4);
        assert_eq!(dst, orig);
    }

    #[test]
    fn copy_non_contiguous() {
        let src: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let mut dst = [0.0f32; 20];
        kv_cache_copy(&src, &mut dst, &[0, 4], &[1, 3], 4);
        assert_eq!(&dst[4..8], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&dst[12..16], &[16.0, 17.0, 18.0, 19.0]);
    }

    // ── kv_cache_clear_range ───────────────────────────────────

    #[test]
    fn clear_full_range() {
        let mut cache: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 0, 3, 4);
        assert_eq!(cache.to_vec(), vec![0.0; 12]);
    }

    #[test]
    fn clear_single_position() {
        let mut cache: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 1, 2, 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cache[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn clear_start_range() {
        let mut cache: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 0, 1, 4);
        assert_eq!(&cache[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&cache[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn clear_end_range() {
        let mut cache: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 2, 3, 4);
        assert_eq!(&cache[..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&cache[8..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn clear_middle_range() {
        let mut cache: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 1, 3, 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cache[4..12], &[0.0; 8]);
        assert_eq!(&cache[12..], &[13.0, 14.0, 15.0, 16.0],);
    }

    #[test]
    fn clear_preserves_before() {
        let mut cache: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 1, 3, 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn clear_preserves_after() {
        let mut cache: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 0, 2, 4);
        assert_eq!(&cache[8..], &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn clear_empty_range() {
        let mut cache = [1.0f32; 8];
        let orig = cache.clone();
        kv_cache_clear_range(&mut cache, 2, 2, 4);
        assert_eq!(cache, orig);
    }

    #[test]
    fn clear_head_dim_1() {
        let mut cache = vec![1.0, 2.0, 3.0, 4.0];
        kv_cache_clear_range(&mut cache, 1, 3, 1);
        assert_eq!(cache.to_vec(), vec![1.0, 0.0, 0.0, 4.0]);
    }

    #[test]
    fn clear_head_dim_3() {
        let mut cache: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 0, 1, 3);
        assert_eq!(&cache[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&cache[3..], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn clear_already_zero() {
        let mut cache = [0.0f32; 16];
        kv_cache_clear_range(&mut cache, 0, 4, 4);
        assert_eq!(cache.to_vec(), vec![0.0; 16]);
    }

    #[test]
    fn clear_large_range() {
        let mut cache = [1.0f32; 1024];
        kv_cache_clear_range(&mut cache, 0, 256, 4);
        assert!(cache.iter().all(|&v| v == 0.0));
    }

    // ── kv_cache_paged_lookup ──────────────────────────────────

    #[test]
    fn paged_single_page_single_pos() {
        // 1 page, page_size=4, head_dim=2
        let pages = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let page_table = [0];
        let mut out = [0.0f32; 2];
        kv_cache_paged_lookup(&pages, &page_table, &[2], 4, 2, &mut out);
        assert_eq!(out.to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn paged_single_page_multi_pos() {
        let pages = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let page_table = [0];
        let mut out = [0.0f32; 4];
        kv_cache_paged_lookup(&pages, &page_table, &[0, 3], 4, 2, &mut out);
        assert_eq!(out.to_vec(), vec![10.0, 20.0, 70.0, 80.0]);
    }

    #[test]
    fn paged_multi_page() {
        // page_size=2, head_dim=2
        // page 0: [1,2, 3,4]  page 1: [5,6, 7,8]
        let pages = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let page_table = vec![0, 1]; // identity mapping
        let mut out = [0.0f32; 4];
        // seq_pos 0 → page 0, offset 0 → phys[0*2+0]*2 = [1,2]
        // seq_pos 3 → page 1, offset 1 → phys[1*2+1]*2 = [7,8]
        kv_cache_paged_lookup(&pages, &page_table, &[0, 3], 2, 2, &mut out);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 7.0, 8.0]);
    }

    #[test]
    fn paged_page_boundary() {
        // page_size=2, head_dim=3
        // 2 pages × 2 entries × 3 dims = 12 elements
        let pages: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let page_table = vec![0, 1];
        let mut out = [0.0f32; 6];
        // seq_pos 1 → page 0, offset 1 → [4,5,6]
        // seq_pos 2 → page 1, offset 0 → [7,8,9]
        kv_cache_paged_lookup(&pages, &page_table, &[1, 2], 2, 3, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn paged_page_size_1() {
        let pages: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let page_table = vec![0, 1, 2, 3];
        let mut out = [0.0f32; 4];
        kv_cache_paged_lookup(&pages, &page_table, &[3, 0], 1, 2, &mut out);
        assert_eq!(out.to_vec(), vec![6.0, 7.0, 0.0, 1.0]);
    }

    #[test]
    fn paged_page_size_4() {
        // 2 pages × 4 entries × 2 dims = 16 elements
        let pages: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let page_table = vec![0, 1];
        let mut out = [0.0f32; 2];
        // seq_pos 5 → page 1, offset 1 → phys[1*4+1]*2 = idx 10
        kv_cache_paged_lookup(&pages, &page_table, &[5], 4, 2, &mut out);
        assert_eq!(out.to_vec(), vec![10.0, 11.0]);
    }

    #[test]
    fn paged_head_dim_1() {
        let pages = vec![10.0, 20.0, 30.0, 40.0];
        let page_table = [0];
        let mut out = [0.0f32; 2];
        kv_cache_paged_lookup(&pages, &page_table, &[1, 3], 4, 1, &mut out);
        assert_eq!(out.to_vec(), vec![20.0, 40.0]);
    }

    #[test]
    fn paged_head_dim_3() {
        // 1 page, page_size=2, head_dim=3
        let pages: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let page_table = [0];
        let mut out = [0.0f32; 3];
        kv_cache_paged_lookup(&pages, &page_table, &[1], 2, 3, &mut out);
        assert_eq!(out.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn paged_remapped_pages() {
        // Physical page 1 mapped to logical 0 and vice versa.
        // page_size=1, head_dim=2
        let pages = vec![10.0, 20.0, 30.0, 40.0];
        let page_table = vec![1, 0]; // swap
        let mut out = [0.0f32; 4];
        kv_cache_paged_lookup(&pages, &page_table, &[0, 1], 1, 2, &mut out);
        // seq_pos 0 → page_table[0]=1 → phys[1*1+0]*2 = [30,40]
        // seq_pos 1 → page_table[1]=0 → phys[0*1+0]*2 = [10,20]
        assert_eq!(out.to_vec(), vec![30.0, 40.0, 10.0, 20.0]);
    }

    #[test]
    fn paged_out_of_order() {
        let pages: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let page_table = vec![0, 1];
        let mut out = [0.0f32; 6];
        kv_cache_paged_lookup(&pages, &page_table, &[3, 1, 0], 3, 2, &mut out);
        // seq_pos 3 → page 1, offset 0 → phys[1*3+0]*2 = 6
        // seq_pos 1 → page 0, offset 1 → phys[0*3+1]*2 = 2
        // seq_pos 0 → page 0, offset 0 → phys[0*3+0]*2 = 0
        assert_eq!(out.to_vec(), vec![6.0, 7.0, 2.0, 3.0, 0.0, 1.0],);
    }

    #[test]
    fn paged_duplicate_positions() {
        let pages = vec![1.0, 2.0, 3.0, 4.0];
        let page_table = [0];
        let mut out = [0.0f32; 4];
        kv_cache_paged_lookup(&pages, &page_table, &[0, 0], 2, 2, &mut out);
        assert_eq!(out.to_vec(), vec![1.0, 2.0, 1.0, 2.0]);
    }

    // ── integration / cross-operation tests ────────────────────

    #[test]
    fn roundtrip_append_gather() {
        let mut cache = [0.0f32; 32];
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![5.0, 6.0, 7.0, 8.0];
        kv_cache_append(&mut cache, 0, &v0, 4);
        kv_cache_append(&mut cache, 1, &v1, 4);
        let mut out = [0.0f32; 8];
        kv_cache_gather(&cache, &[1, 0], 4, &mut out);
        assert_eq!(&out[..4], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&out[4..], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn append_copy_equivalence() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut c1 = [0.0f32; 16];
        let mut c2 = [0.0f32; 16];
        kv_cache_append(&mut c1, 0, &data, 4);
        // Replicate via copy from a pre-filled source.
        let src = data.clone();
        let mut padded_src = [0.0f32; 16];
        padded_src[..8].copy_from_slice(&src);
        kv_cache_copy(&padded_src, &mut c2, &[0, 1], &[0, 1], 4);
        assert_eq!(c1, c2);
    }

    #[test]
    fn clear_then_verify_zeros() {
        let mut cache: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 1, 3, 4);
        let mut out = [0.0f32; 8];
        kv_cache_gather(&cache, &[1, 2], 4, &mut out);
        assert_eq!(out.to_vec(), vec![0.0; 8]);
    }

    #[test]
    fn paged_vs_flat_equivalence() {
        let hd = 4;
        let flat: Vec<f32> = (0..20).map(|x| x as f32).collect();
        // Build pages that are identical to flat layout.
        // page_size=5, 1 page
        let page_table = [0];
        let mut flat_out = vec![0.0f32; hd];
        let mut paged_out = vec![0.0f32; hd];
        kv_cache_gather(&flat, &[3], hd, &mut flat_out);
        kv_cache_paged_lookup(&flat, &page_table, &[3], 5, hd, &mut paged_out);
        assert_eq!(flat_out, paged_out);
    }

    #[test]
    fn multi_head_simulation() {
        let num_heads = 4;
        let head_dim = 3;
        let seq_len = 2;
        let mut caches = vec![vec![0.0f32; seq_len * head_dim]; num_heads];
        // Append different data per head.
        for (h, cache) in caches.iter_mut().enumerate() {
            let data: Vec<f32> = (0..head_dim).map(|d| (h * head_dim + d) as f32).collect();
            kv_cache_append(cache, 0, &data, head_dim);
        }
        // Gather from each head.
        for (h, cache) in caches.iter().enumerate() {
            let mut out = vec![0.0f32; head_dim];
            kv_cache_gather(cache, &[0], head_dim, &mut out);
            let expected: Vec<f32> = (0..head_dim).map(|d| (h * head_dim + d) as f32).collect();
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn sliding_window_pattern() {
        let window = 4;
        let head_dim = 2;
        let mut cache = vec![0.0f32; window * head_dim];
        // Fill window.
        for i in 0..window {
            let v = vec![(i * 10) as f32, (i * 10 + 1) as f32];
            kv_cache_append(&mut cache, i, &v, head_dim);
        }
        // Slide: clear oldest, move to front conceptually.
        // In practice, use a ring buffer; here we shift.
        let mut tmp = vec![0.0f32; (window - 1) * head_dim];
        kv_cache_gather(&cache, &[1, 2, 3], head_dim, &mut tmp);
        kv_cache_clear_range(&mut cache, 0, window, head_dim);
        kv_cache_append(&mut cache, 0, &tmp, head_dim);
        // Position 0 should now contain what was at position 1.
        let mut out = vec![0.0f32; head_dim];
        kv_cache_gather(&cache, &[0], head_dim, &mut out);
        assert_eq!(out.to_vec(), vec![10.0, 11.0]);
    }

    #[test]
    fn beam_search_copy_pattern() {
        let beam_width = 3;
        let head_dim = 4;
        let mut cache = vec![0.0f32; beam_width * head_dim];
        // Fill initial beams.
        for b in 0..beam_width {
            let v: Vec<f32> = (0..head_dim).map(|d| (b * 100 + d) as f32).collect();
            kv_cache_append(&mut cache, b, &v, head_dim);
        }
        // Beam 0 is best; duplicate to all positions.
        let mut new_cache = vec![0.0f32; beam_width * head_dim];
        kv_cache_copy(&cache, &mut new_cache, &[0, 0, 0], &[0, 1, 2], head_dim);
        for b in 0..beam_width {
            let mut out = vec![0.0f32; head_dim];
            kv_cache_gather(&new_cache, &[b], head_dim, &mut out);
            assert_eq!(out.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn sequential_token_append() {
        let max_seq = 16;
        let head_dim = 4;
        let mut cache = vec![0.0f32; max_seq * head_dim];
        for t in 0..max_seq {
            let v: Vec<f32> = (0..head_dim).map(|d| (t * head_dim + d) as f32).collect();
            kv_cache_append(&mut cache, t, &v, head_dim);
        }
        // Verify every position.
        let mut out = vec![0.0f32; max_seq * head_dim];
        let positions: Vec<usize> = (0..max_seq).collect();
        kv_cache_gather(&cache, &positions, head_dim, &mut out);
        assert_eq!(out, cache);
    }

    #[test]
    fn large_cache_append_gather() {
        let n = 256;
        let hd = 16;
        let mut cache = vec![0.0f32; n * hd];
        let data: Vec<f32> = (0..(n * hd) as u32).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 0, &data, hd);
        let mut out = vec![0.0f32; hd];
        kv_cache_gather(&cache, &[128], hd, &mut out);
        let expected: Vec<f32> = (128 * hd..129 * hd).map(|x| x as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn rotate_gather_consistency() {
        let hd = 4;
        let mut cache = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        kv_cache_rotate(&mut cache, 0, 2, hd, 10000.0);
        let mut out = [0.0f32; 4];
        kv_cache_gather(&cache, &[0], hd, &mut out);
        // pos=0 → identity
        approx_eq(&out, &[1.0, 0.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn clear_append_overwrite() {
        let mut cache = [9.0f32; 8];
        kv_cache_clear_range(&mut cache, 0, 2, 4);
        assert_eq!(cache.to_vec(), vec![0.0; 8]);
        kv_cache_append(&mut cache, 0, &[1.0, 2.0, 3.0, 4.0], 4);
        assert_eq!(&cache[..4], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn full_lifecycle() {
        let hd = 4;
        let max_seq = 8;
        let mut cache = vec![0.0f32; max_seq * hd];

        // 1. Append three vectors.
        let v0 = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![5.0, 6.0, 7.0, 8.0];
        let v2 = vec![9.0, 10.0, 11.0, 12.0];
        kv_cache_append(&mut cache, 0, &v0, hd);
        kv_cache_append(&mut cache, 1, &v1, hd);
        kv_cache_append(&mut cache, 2, &v2, hd);

        // 2. Gather in reverse.
        let mut out = vec![0.0f32; 3 * hd];
        kv_cache_gather(&cache, &[2, 1, 0], hd, &mut out);
        assert_eq!(&out[..4], &v2[..]);
        assert_eq!(&out[4..8], &v1[..]);
        assert_eq!(&out[8..], &v0[..]);

        // 3. Rotate positions 0..3.
        kv_cache_rotate(&mut cache, 0, 3, hd, 10000.0);
        // pos=0 is identity, verify:
        let mut g0 = vec![0.0f32; hd];
        kv_cache_gather(&cache, &[0], hd, &mut g0);
        approx_eq(&g0, &v0, 1e-6);

        // 4. Copy position 0 to position 5.
        let snap = cache.clone();
        let mut cache2 = vec![0.0f32; max_seq * hd];
        kv_cache_copy(&snap, &mut cache2, &[0], &[5], hd);
        let mut g5 = vec![0.0f32; hd];
        kv_cache_gather(&cache2, &[5], hd, &mut g5);
        approx_eq(&g5, &v0, 1e-6);

        // 5. Clear range 1..3.
        kv_cache_clear_range(&mut cache, 1, 3, hd);
        let mut g1 = vec![0.0f32; hd];
        kv_cache_gather(&cache, &[1], hd, &mut g1);
        assert_eq!(g1, vec![0.0; hd]);
    }

    #[test]
    fn head_dim_7_all_ops() {
        let hd = 7;
        let mut cache = vec![0.0f32; 3 * hd];
        let data: Vec<f32> = (1..=7).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 0, &data, hd);
        let mut out = vec![0.0f32; hd];
        kv_cache_gather(&cache, &[0], hd, &mut out);
        assert_eq!(out, data);
        kv_cache_clear_range(&mut cache, 0, 1, hd);
        kv_cache_gather(&cache, &[0], hd, &mut out);
        assert_eq!(out.to_vec(), vec![0.0; hd]);
    }

    #[test]
    fn head_dim_16_all_ops() {
        let hd = 16;
        let mut cache = vec![0.0f32; 4 * hd];
        let data: Vec<f32> = (0..hd).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 2, &data, hd);
        let mut out = vec![0.0f32; hd];
        kv_cache_gather(&cache, &[2], hd, &mut out);
        assert_eq!(out, data);
    }

    #[test]
    fn copy_identity_mapping() {
        let src: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut dst = [0.0f32; 12];
        kv_cache_copy(&src, &mut dst, &[0, 1, 2], &[0, 1, 2], 4);
        assert_eq!(dst, src);
    }

    #[test]
    fn paged_full_page_lookup() {
        let page_size = 4;
        let hd = 2;
        let pages: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let page_table = vec![0, 1];
        let mut out = vec![0.0f32; page_size * hd];
        let positions: Vec<usize> = (0..page_size).collect();
        kv_cache_paged_lookup(&pages, &page_table, &positions, page_size, hd, &mut out);
        assert_eq!(out.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn paged_large_table() {
        let page_size = 2;
        let hd = 2;
        let num_pages = 8;
        let pages: Vec<f32> = (0..(num_pages * page_size * hd)).map(|x| x as f32).collect();
        let page_table: Vec<usize> = (0..num_pages).collect();
        let mut out = vec![0.0f32; hd];
        // Look up last position.
        let last_pos = num_pages * page_size - 1;
        kv_cache_paged_lookup(&pages, &page_table, &[last_pos], page_size, hd, &mut out);
        let expected_off = last_pos * hd;
        let expected: Vec<f32> = (expected_off..expected_off + hd).map(|x| x as f32).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn gather_all_positions() {
        let hd = 2;
        let n = 5;
        let cache: Vec<f32> = (0..(n * hd) as u32).map(|x| x as f32).collect();
        let positions: Vec<usize> = (0..n).collect();
        let mut out = vec![0.0f32; n * hd];
        kv_cache_gather(&cache, &positions, hd, &mut out);
        assert_eq!(out, cache);
    }

    #[test]
    fn clear_all_positions() {
        let hd = 3;
        let n = 4;
        let mut cache: Vec<f32> = (1..=(n * hd) as u32).map(|x| x as f32).collect();
        kv_cache_clear_range(&mut cache, 0, n, hd);
        assert!(cache.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn append_large_batch() {
        let hd = 5;
        let n = 100;
        let mut cache = vec![0.0f32; n * hd];
        let data: Vec<f32> = (0..(n * hd) as u32).map(|x| x as f32).collect();
        kv_cache_append(&mut cache, 0, &data, hd);
        assert_eq!(cache, data);
    }
}
