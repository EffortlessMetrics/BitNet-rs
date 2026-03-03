//! NEON-optimized RoPE (Rotary Position Embedding) kernels for Apple Silicon.

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

// -- Constants ---------------------------------------------------------------

/// Default rotation base frequency (original RoPE paper).
#[cfg(test)]
const DEFAULT_BASE: f32 = 10_000.0;

// -- Scalar reference implementations ----------------------------------------

/// Scalar: precompute separate cos/sin frequency tables.
///
/// Layout: `table[pos * half_dim + i]` for position `pos` and pair index `i`.
pub fn scalar_precompute_freqs(
    dim: usize,
    max_seq: usize,
    base: f32,
    scaling_factor: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;
    let mut cos_table = Vec::with_capacity(max_seq * half_dim);
    let mut sin_table = Vec::with_capacity(max_seq * half_dim);
    for pos in 0..max_seq {
        for i in 0..half_dim {
            let exponent = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exponent) * scaling_factor;
            let angle = pos as f32 * theta;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }
    (cos_table, sin_table)
}

/// Scalar: apply standard interleaved RoPE rotation in-place on a single head.
///
/// Pairs `(data[2i], data[2i+1])` are rotated by the angle for position `pos`.
pub fn scalar_rope_apply(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half_dim = dim / 2;
    let off = pos * half_dim;
    for i in 0..half_dim {
        let c = cos_table[off + i];
        let s = sin_table[off + i];
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * c - x1 * s;
        data[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// Scalar: half-rotary RoPE -- rotate first half of dimensions, pass through rest.
pub fn scalar_rope_apply_half(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let rotary_dim = dim / 2;
    let half_rotary = rotary_dim / 2;
    let off = pos * (dim / 2);
    for i in 0..half_rotary {
        let c = cos_table[off + i];
        let s = sin_table[off + i];
        let x0 = data[2 * i];
        let x1 = data[2 * i + 1];
        data[2 * i] = x0 * c - x1 * s;
        data[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// Scalar: NeoX-style RoPE (split halves rather than interleaved).
///
/// Pairs are `(data[i], data[i + half_dim])` for `i in 0..half_dim`.
pub fn scalar_rope_neox(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half_dim = dim / 2;
    let off = pos * half_dim;
    for i in 0..half_dim {
        let c = cos_table[off + i];
        let s = sin_table[off + i];
        let x0 = data[i];
        let x1 = data[i + half_dim];
        data[i] = x0 * c - x1 * s;
        data[i + half_dim] = x0 * s + x1 * c;
    }
}

/// Scalar: batched RoPE across `[seq_len, num_heads, dim]`.
pub fn scalar_rope_batched(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    start_pos: usize,
    seq_len: usize,
) {
    for s in 0..seq_len {
        let pos = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * dim;
            scalar_rope_apply(&mut data[offset..offset + dim], cos_table, sin_table, dim, pos);
        }
    }
}

/// Scalar: apply RoPE using a precomputed cache (pos-indexed slices).
pub fn scalar_rope_apply_with_cache(
    data: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    dim: usize,
    pos: usize,
) {
    scalar_rope_apply(data, cos_cache, sin_cache, dim, pos);
}

// -- NEON implementations ----------------------------------------------------

/// NEON: precompute separate cos/sin frequency tables.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_precompute_freqs(
    dim: usize,
    max_seq: usize,
    base: f32,
    scaling_factor: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half_dim = dim / 2;
    let total = max_seq * half_dim;
    let mut cos_table = Vec::with_capacity(total);
    let mut sin_table = Vec::with_capacity(total);
    for pos in 0..max_seq {
        for i in 0..half_dim {
            let exponent = -(2.0 * i as f32) / dim as f32;
            let theta = base.powf(exponent) * scaling_factor;
            let angle = pos as f32 * theta;
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }
    (cos_table, sin_table)
}

/// NEON: apply standard interleaved RoPE rotation in-place.
///
/// Processes 4 floats (2 rotation pairs) per iteration via `float32x4_t`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half_dim = dim / 2;
    let table_offset = pos * half_dim;

    let sign_mask = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());

    let chunks = half_dim / 2;
    for c in 0..chunks {
        let data_idx = c * 4;
        let table_idx = table_offset + c * 2;

        let vals = vld1q_f32(data.as_ptr().add(data_idx));
        let swapped = vrev64q_f32(vals);

        let c0 = *cos_table.get_unchecked(table_idx);
        let c1 = *cos_table.get_unchecked(table_idx + 1);
        let s0 = *sin_table.get_unchecked(table_idx);
        let s1 = *sin_table.get_unchecked(table_idx + 1);

        let cos_exp = vld1q_f32([c0, c0, c1, c1].as_ptr());
        let sin_exp = vld1q_f32([s0, s0, s1, s1].as_ptr());

        let term1 = vmulq_f32(vals, cos_exp);
        let term2 = vmulq_f32(vmulq_f32(swapped, sign_mask), sin_exp);
        let rotated = vaddq_f32(term1, term2);

        vst1q_f32(data.as_mut_ptr().add(data_idx), rotated);
    }

    // Scalar tail for remaining pair when half_dim is odd.
    let processed_pairs = chunks * 2;
    for i in processed_pairs..half_dim {
        let idx = i * 2;
        let cv = *cos_table.get_unchecked(table_offset + i);
        let sv = *sin_table.get_unchecked(table_offset + i);
        let x0 = data[idx];
        let x1 = data[idx + 1];
        data[idx] = x0 * cv - x1 * sv;
        data[idx + 1] = x0 * sv + x1 * cv;
    }
}

/// NEON: half-rotary RoPE -- rotate first half, pass through second half.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply_half(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let rotary_dim = dim / 2;
    let half_rotary = rotary_dim / 2;
    let table_offset = pos * (dim / 2);

    let sign_mask = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());

    let chunks = half_rotary / 2;
    for c in 0..chunks {
        let data_idx = c * 4;
        let table_idx = table_offset + c * 2;

        let vals = vld1q_f32(data.as_ptr().add(data_idx));
        let swapped = vrev64q_f32(vals);

        let c0 = *cos_table.get_unchecked(table_idx);
        let c1 = *cos_table.get_unchecked(table_idx + 1);
        let s0 = *sin_table.get_unchecked(table_idx);
        let s1 = *sin_table.get_unchecked(table_idx + 1);

        let cos_exp = vld1q_f32([c0, c0, c1, c1].as_ptr());
        let sin_exp = vld1q_f32([s0, s0, s1, s1].as_ptr());

        let term1 = vmulq_f32(vals, cos_exp);
        let term2 = vmulq_f32(vmulq_f32(swapped, sign_mask), sin_exp);
        let rotated = vaddq_f32(term1, term2);

        vst1q_f32(data.as_mut_ptr().add(data_idx), rotated);
    }

    // Scalar tail.
    let processed_pairs = chunks * 2;
    for i in processed_pairs..half_rotary {
        let idx = i * 2;
        let cv = *cos_table.get_unchecked(table_offset + i);
        let sv = *sin_table.get_unchecked(table_offset + i);
        let x0 = data[idx];
        let x1 = data[idx + 1];
        data[idx] = x0 * cv - x1 * sv;
        data[idx + 1] = x0 * sv + x1 * cv;
    }
}

/// NEON: NeoX-style RoPE (split halves instead of interleaved pairs).
///
/// Pairs are `(data[i], data[i + half_dim])` for `i in 0..half_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_neox(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    pos: usize,
) {
    let half_dim = dim / 2;
    let off = pos * half_dim;

    // Process 4 pairs per iteration.
    let chunks = half_dim / 4;
    for c in 0..chunks {
        let i = c * 4;
        let first = vld1q_f32(data.as_ptr().add(i));
        let second = vld1q_f32(data.as_ptr().add(i + half_dim));

        let cos_v = vld1q_f32(cos_table.as_ptr().add(off + i));
        let sin_v = vld1q_f32(sin_table.as_ptr().add(off + i));

        let first_out = vsubq_f32(vmulq_f32(first, cos_v), vmulq_f32(second, sin_v));
        let second_out = vaddq_f32(vmulq_f32(first, sin_v), vmulq_f32(second, cos_v));

        vst1q_f32(data.as_mut_ptr().add(i), first_out);
        vst1q_f32(data.as_mut_ptr().add(i + half_dim), second_out);
    }

    // Scalar tail.
    for i in (chunks * 4)..half_dim {
        let cv = *cos_table.get_unchecked(off + i);
        let sv = *sin_table.get_unchecked(off + i);
        let x0 = data[i];
        let x1 = data[i + half_dim];
        data[i] = x0 * cv - x1 * sv;
        data[i + half_dim] = x0 * sv + x1 * cv;
    }
}

/// NEON: batched RoPE across `[seq_len, num_heads, dim]`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_batched(
    data: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    dim: usize,
    num_heads: usize,
    start_pos: usize,
    seq_len: usize,
) {
    for s in 0..seq_len {
        let pos = start_pos + s;
        for h in 0..num_heads {
            let offset = (s * num_heads + h) * dim;
            neon_rope_apply(&mut data[offset..offset + dim], cos_table, sin_table, dim, pos);
        }
    }
}

/// NEON: apply RoPE using a precomputed sin/cos cache.
///
/// Functionally identical to [`neon_rope_apply`] -- the cache shares the same
/// `table[pos * half_dim + i]` layout.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply_with_cache(
    data: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    dim: usize,
    pos: usize,
) {
    neon_rope_apply(data, cos_cache, sin_cache, dim, pos);
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(diff <= tol, "index {i}: {x} vs {y} (diff {diff} > tol {tol})");
        }
    }

    fn make_data(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32 + 1.0) * 0.1).collect()
    }

    fn make_data_seeded(len: usize, seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(len);
        let mut s = seed;
        for _ in 0..len {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((s >> 16) as f32 / 32768.0) - 1.0;
            v.push(f);
        }
        v
    }

    // -- Basic correctness vs scalar reference (15+) -------------------------

    #[test]
    fn test_scalar_precompute_basic() {
        let (cos, sin) = scalar_precompute_freqs(4, 2, DEFAULT_BASE, 1.0);
        assert_eq!(cos.len(), 4);
        assert_eq!(sin.len(), 4);
        assert!((cos[0] - 1.0).abs() < EPS);
        assert!(sin[0].abs() < EPS);
    }

    #[test]
    fn test_scalar_apply_identity_at_pos0() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 1, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply(&mut data, &cos, &sin, dim, 0);
        assert_close(&data, &orig, EPS);
    }

    #[test]
    fn test_scalar_apply_dim4() {
        let dim = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = vec![1.0, 0.0, 1.0, 0.0];
        scalar_rope_apply(&mut data, &cos, &sin, dim, 1);
        assert!((data[0] - 1.0).abs() > 1e-7 || data[1].abs() > 1e-7);
    }

    #[test]
    fn test_scalar_neox_identity_at_pos0() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 1, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_neox(&mut data, &cos, &sin, dim, 0);
        assert_close(&data, &orig, EPS);
    }

    #[test]
    fn test_scalar_half_identity_at_pos0() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 1, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 0);
        assert_close(&data, &orig, EPS);
    }

    #[test]
    fn test_scalar_batched_single_head_single_seq() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data_batch = make_data(dim);
        let mut data_single = data_batch.clone();
        scalar_rope_batched(&mut data_batch, &cos, &sin, dim, 1, 2, 1);
        scalar_rope_apply(&mut data_single, &cos, &sin, dim, 2);
        assert_close(&data_batch, &data_single, EPS);
    }

    #[test]
    fn test_scalar_with_cache_matches_apply() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut d1 = make_data(dim);
        let mut d2 = d1.clone();
        scalar_rope_apply(&mut d1, &cos, &sin, dim, 3);
        scalar_rope_apply_with_cache(&mut d2, &cos, &sin, dim, 3);
        assert_close(&d1, &d2, EPS);
    }

    #[test]
    fn test_scalar_rotation_is_unitary() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
        let data = make_data_seeded(dim, 42);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        scalar_rope_apply(&mut rotated, &cos, &sin, dim, 5);
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!((norm_before - norm_after).abs() < 1e-4, "rotation should preserve norm");
    }

    #[test]
    fn test_scalar_neox_rotation_is_unitary() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
        let data = make_data_seeded(dim, 77);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        scalar_rope_neox(&mut rotated, &cos, &sin, dim, 7);
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!((norm_before - norm_after).abs() < 1e-4);
    }

    #[test]
    fn test_scalar_apply_multiple_positions() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let base = make_data(dim);
        let mut d1 = base.clone();
        let mut d2 = base.clone();
        scalar_rope_apply(&mut d1, &cos, &sin, dim, 3);
        scalar_rope_apply(&mut d2, &cos, &sin, dim, 5);
        let differs = d1.iter().zip(d2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs, "different positions should give different rotations");
    }

    #[test]
    fn test_scalar_scaling_factor() {
        let (cos1, _) = scalar_precompute_freqs(8, 4, DEFAULT_BASE, 1.0);
        let (cos2, _) = scalar_precompute_freqs(8, 4, DEFAULT_BASE, 2.0);
        let differs = cos1.iter().zip(cos2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs);
    }

    #[test]
    fn test_scalar_base_frequency_effect() {
        let (cos1, _) = scalar_precompute_freqs(8, 4, 10_000.0, 1.0);
        let (cos2, _) = scalar_precompute_freqs(8, 4, 1_000.0, 1.0);
        let differs = cos1.iter().zip(cos2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs);
    }

    #[test]
    fn test_scalar_precompute_symmetry() {
        let (cos, sin) = scalar_precompute_freqs(8, 4, DEFAULT_BASE, 1.0);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!((sum - 1.0).abs() < 1e-5, "pythagorean identity at index {i}");
        }
    }

    #[test]
    fn test_scalar_half_preserves_second_half() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 2);
        let rotary_dim = dim / 2;
        assert_close(&data[rotary_dim..], &orig[rotary_dim..], EPS);
    }

    #[test]
    fn test_scalar_neox_vs_standard_differ() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut d_std = make_data(dim);
        let mut d_neox = d_std.clone();
        scalar_rope_apply(&mut d_std, &cos, &sin, dim, 2);
        scalar_rope_neox(&mut d_neox, &cos, &sin, dim, 2);
        let differs = d_std.iter().zip(d_neox.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs, "standard and neox should produce different results");
    }

    // -- Various head dimensions (15+) ---------------------------------------

    #[test]
    fn test_scalar_dim32() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        scalar_rope_apply(&mut data, &cos, &sin, dim, 2);
        assert!(data.iter().map(|x| x * x).sum::<f32>() > 0.0);
    }

    #[test]
    fn test_scalar_dim64() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        scalar_rope_apply(&mut data, &cos, &sin, dim, 3);
        assert!(data.iter().map(|x| x * x).sum::<f32>() > 0.0);
    }

    #[test]
    fn test_scalar_dim128() {
        let dim = 128;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 100);
        let orig_norm: f32 = data.iter().map(|x| x * x).sum();
        scalar_rope_apply(&mut data, &cos, &sin, dim, 1);
        let new_norm: f32 = data.iter().map(|x| x * x).sum();
        assert!((orig_norm - new_norm).abs() < 1e-3);
    }

    #[test]
    fn test_scalar_dim256() {
        let dim = 256;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 200);
        let orig_norm: f32 = data.iter().map(|x| x * x).sum();
        scalar_rope_apply(&mut data, &cos, &sin, dim, 2);
        let new_norm: f32 = data.iter().map(|x| x * x).sum();
        assert!((orig_norm - new_norm).abs() < 1e-2);
    }

    #[test]
    fn test_scalar_neox_dim32() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        scalar_rope_neox(&mut data, &cos, &sin, dim, 1);
        assert!(data.iter().map(|x| x * x).sum::<f32>() > 0.0);
    }

    #[test]
    fn test_scalar_neox_dim64() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 55);
        let orig_norm: f32 = data.iter().map(|x| x * x).sum();
        scalar_rope_neox(&mut data, &cos, &sin, dim, 5);
        let new_norm: f32 = data.iter().map(|x| x * x).sum();
        assert!((orig_norm - new_norm).abs() < 1e-3);
    }

    #[test]
    fn test_scalar_neox_dim128() {
        let dim = 128;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 99);
        scalar_rope_neox(&mut data, &cos, &sin, dim, 2);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_scalar_neox_dim256() {
        let dim = 256;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 123);
        scalar_rope_neox(&mut data, &cos, &sin, dim, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_scalar_half_dim32() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 2);
        assert_close(&data[16..], &orig[16..], EPS);
    }

    #[test]
    fn test_scalar_half_dim64() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 1);
        assert_close(&data[32..], &orig[32..], EPS);
    }

    #[test]
    fn test_scalar_half_dim128() {
        let dim = 128;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 10);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 3);
        assert_close(&data[64..], &orig[64..], EPS);
    }

    #[test]
    fn test_scalar_half_dim256() {
        let dim = 256;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 20);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 2);
        assert_close(&data[128..], &orig[128..], EPS);
    }

    #[test]
    fn test_scalar_batched_dim32_4heads() {
        let dim = 32;
        let num_heads = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * num_heads);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_scalar_batched_dim128_2heads() {
        let dim = 128;
        let num_heads = 2;
        let seq = 3;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * num_heads * seq);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 1, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    // -- Sequence length variations (15+) ------------------------------------

    #[test]
    fn test_seq_len_1() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 0, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_4() {
        let dim = 8;
        let num_heads = 2;
        let seq = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * num_heads * seq);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_16() {
        let dim = 16;
        let seq = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 32, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * seq);
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_128() {
        let dim = 32;
        let seq = 128;
        let (cos, sin) = scalar_precompute_freqs(dim, 256, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * seq);
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_512() {
        let dim = 64;
        let seq = 512;
        let (cos, sin) = scalar_precompute_freqs(dim, 1024, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim * seq, 333);
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_1_with_offset() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 128, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 100, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_4_multihead() {
        let dim = 16;
        let seq = 4;
        let num_heads = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * num_heads * seq);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_16_neox() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 32, DEFAULT_BASE, 1.0);
        for pos in 0..16 {
            let mut data = make_data(dim);
            scalar_rope_neox(&mut data, &cos, &sin, dim, pos);
            assert!(data.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_seq_positions_differ() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let base = make_data(dim);
        let mut results = Vec::new();
        for pos in 0..8 {
            let mut d = base.clone();
            scalar_rope_apply(&mut d, &cos, &sin, dim, pos);
            results.push(d);
        }
        for i in 1..8 {
            for j in (i + 1)..8 {
                let differs =
                    results[i].iter().zip(results[j].iter()).any(|(a, b)| (a - b).abs() > 1e-7);
                assert!(differs, "pos {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn test_seq_128_norm_preservation() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 256, DEFAULT_BASE, 1.0);
        for pos in [0, 1, 50, 127, 200] {
            let data = make_data_seeded(dim, pos as u32 + 1);
            let orig_norm: f32 = data.iter().map(|x| x * x).sum();
            let mut rotated = data;
            scalar_rope_apply(&mut rotated, &cos, &sin, dim, pos);
            let new_norm: f32 = rotated.iter().map(|x| x * x).sum();
            assert!((orig_norm - new_norm).abs() < 1e-3, "norm changed at pos {pos}");
        }
    }

    #[test]
    fn test_seq_512_dim64_batch() {
        let dim = 64;
        let num_heads = 2;
        let seq = 512;
        let (cos, sin) = scalar_precompute_freqs(dim, 1024, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim * num_heads * seq, 777);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_seq_len_1_half_rotary() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 1);
        assert_close(&data[8..], &orig[8..], EPS);
    }

    #[test]
    fn test_seq_len_128_half_rotary() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 256, DEFAULT_BASE, 1.0);
        for pos in [0, 1, 64, 127] {
            let mut data = make_data_seeded(dim, pos as u32 + 10);
            let orig = data.clone();
            scalar_rope_apply_half(&mut data, &cos, &sin, dim, pos);
            assert_close(&data[32..], &orig[32..], EPS);
        }
    }

    #[test]
    fn test_seq_len_512_neox() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 1024, DEFAULT_BASE, 1.0);
        for pos in [0, 1, 256, 511] {
            let mut data = make_data_seeded(dim, pos as u32 + 50);
            scalar_rope_neox(&mut data, &cos, &sin, dim, pos);
            assert!(data.iter().all(|x| x.is_finite()));
        }
    }

    // -- Batched operations (10+) --------------------------------------------

    #[test]
    fn test_batched_1head_1seq() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let mut single = data.clone();
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 2, 1);
        scalar_rope_apply(&mut single, &cos, &sin, dim, 2);
        assert_close(&data, &single, EPS);
    }

    #[test]
    fn test_batched_4heads_1seq() {
        let dim = 16;
        let num_heads = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim * num_heads);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 1, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batched_matches_individual() {
        let dim = 8;
        let num_heads = 2;
        let seq = 3;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut batch_data = make_data(dim * num_heads * seq);
        let orig = batch_data.clone();
        scalar_rope_batched(&mut batch_data, &cos, &sin, dim, num_heads, 0, seq);

        let mut individual = orig;
        for s in 0..seq {
            for h in 0..num_heads {
                let off = (s * num_heads + h) * dim;
                scalar_rope_apply(&mut individual[off..off + dim], &cos, &sin, dim, s);
            }
        }
        assert_close(&batch_data, &individual, EPS);
    }

    #[test]
    fn test_batched_with_start_pos() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
        let input = make_data(dim * 2);
        let mut data = input.clone();
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 5, 2);
        let mut d5 = input[..dim].to_vec();
        let mut d6 = input[dim..].to_vec();
        scalar_rope_apply(&mut d5, &cos, &sin, dim, 5);
        scalar_rope_apply(&mut d6, &cos, &sin, dim, 6);
        assert_close(&data[..dim], &d5, EPS);
        assert_close(&data[dim..], &d6, EPS);
    }

    #[test]
    fn test_batched_8heads_seq4() {
        let dim = 32;
        let num_heads = 8;
        let seq = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim * num_heads * seq, 42);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batched_preserves_independence() {
        let dim = 16;
        let num_heads = 2;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let head_data = make_data(dim);
        let mut data = Vec::new();
        data.extend_from_slice(&head_data);
        data.extend_from_slice(&head_data);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 1, 1);
        assert_close(&data[..dim], &data[dim..], EPS);
    }

    #[test]
    fn test_batched_large_seq() {
        let dim = 64;
        let num_heads = 4;
        let seq = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 64, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim * num_heads * seq, 999);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batched_single_element() {
        let dim = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 2, DEFAULT_BASE, 1.0);
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        scalar_rope_batched(&mut data, &cos, &sin, dim, 1, 0, 1);
        assert_close(&data, &[1.0, 0.0, 0.0, 1.0], EPS);
    }

    #[test]
    fn test_batched_16heads_dim64() {
        let dim = 64;
        let num_heads = 16;
        let seq = 2;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim * num_heads * seq, 888);
        scalar_rope_batched(&mut data, &cos, &sin, dim, num_heads, 0, seq);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batched_norm_preserved() {
        let dim = 32;
        let num_heads = 4;
        let seq = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let data = make_data_seeded(dim * num_heads * seq, 222);
        let orig_norm: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        scalar_rope_batched(&mut rotated, &cos, &sin, dim, num_heads, 0, seq);
        let new_norm: f32 = rotated.iter().map(|x| x * x).sum();
        assert!((orig_norm - new_norm).abs() / orig_norm < 1e-4);
    }

    // -- NeoX vs standard rotation style (10+) ------------------------------

    #[test]
    fn test_neox_identity_pos0() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 2, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_neox(&mut data, &cos, &sin, dim, 0);
        assert_close(&data, &orig, EPS);
    }

    #[test]
    fn test_neox_rotation_unitary() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let data = make_data_seeded(dim, 50);
        let n0: f32 = data.iter().map(|x| x * x).sum();
        let mut r = data;
        scalar_rope_neox(&mut r, &cos, &sin, dim, 3);
        let n1: f32 = r.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-3);
    }

    #[test]
    fn test_neox_differs_from_standard() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut std_data = make_data(dim);
        let mut neox_data = std_data.clone();
        scalar_rope_apply(&mut std_data, &cos, &sin, dim, 2);
        scalar_rope_neox(&mut neox_data, &cos, &sin, dim, 2);
        let differs = std_data.iter().zip(neox_data.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs);
    }

    #[test]
    fn test_neox_positions_differ() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let base = make_data(dim);
        let mut r1 = base.clone();
        let mut r2 = base.clone();
        scalar_rope_neox(&mut r1, &cos, &sin, dim, 1);
        scalar_rope_neox(&mut r2, &cos, &sin, dim, 5);
        let differs = r1.iter().zip(r2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs);
    }

    #[test]
    fn test_neox_dim64() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 44);
        scalar_rope_neox(&mut data, &cos, &sin, dim, 3);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_neox_dim128() {
        let dim = 128;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data_seeded(dim, 66);
        scalar_rope_neox(&mut data, &cos, &sin, dim, 2);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_neox_pythagorean() {
        let dim = 16;
        let half = dim / 2;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut data = vec![0.0f32; dim];
        data[0] = 1.0;
        scalar_rope_neox(&mut data, &cos, &sin, dim, 3);
        let sum = data[0] * data[0] + data[half] * data[half];
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_neox_all_positions_finite() {
        let dim = 32;
        let max_seq = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, max_seq, DEFAULT_BASE, 1.0);
        for pos in 0..max_seq {
            let mut data = make_data(dim);
            scalar_rope_neox(&mut data, &cos, &sin, dim, pos);
            assert!(data.iter().all(|x| x.is_finite()), "NaN at pos {pos}");
        }
    }

    #[test]
    fn test_neox_vs_standard_same_base() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let base = make_data(dim);
        let mut s = base.clone();
        let mut n = base.clone();
        scalar_rope_apply(&mut s, &cos, &sin, dim, 1);
        scalar_rope_neox(&mut n, &cos, &sin, dim, 1);
        let sn: f32 = s.iter().map(|x| x * x).sum();
        let nn: f32 = n.iter().map(|x| x * x).sum();
        let orig_n: f32 = base.iter().map(|x| x * x).sum();
        assert!((sn - orig_n).abs() < 1e-4);
        assert!((nn - orig_n).abs() < 1e-4);
    }

    #[test]
    fn test_neox_symmetry_zero_input() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = vec![0.0f32; dim];
        scalar_rope_neox(&mut data, &cos, &sin, dim, 2);
        assert!(data.iter().all(|x| x.abs() < EPS), "zero input -> zero output");
    }

    // -- Frequency precomputation correctness (10+) --------------------------

    #[test]
    fn test_freq_pos0_is_identity() {
        let dim = 16;
        let half = dim / 2;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        for i in 0..half {
            assert!((cos[i] - 1.0).abs() < EPS, "cos[{i}] at pos=0 should be 1");
            assert!(sin[i].abs() < EPS, "sin[{i}] at pos=0 should be 0");
        }
    }

    #[test]
    fn test_freq_table_length() {
        let dim = 32;
        let max_seq = 10;
        let (cos, sin) = scalar_precompute_freqs(dim, max_seq, DEFAULT_BASE, 1.0);
        let expected = max_seq * (dim / 2);
        assert_eq!(cos.len(), expected);
        assert_eq!(sin.len(), expected);
    }

    #[test]
    fn test_freq_pythagorean_identity() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!((sum - 1.0).abs() < 1e-5, "pythagorean at {i}");
        }
    }

    #[test]
    fn test_freq_monotone_angle_growth() {
        let dim = 8;
        let half = dim / 2;
        let (_, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        assert!(sin[half] > sin[0]);
    }

    #[test]
    fn test_freq_different_bases() {
        let (c1, _) = scalar_precompute_freqs(16, 4, 10_000.0, 1.0);
        let (c2, _) = scalar_precompute_freqs(16, 4, 500_000.0, 1.0);
        let diff = c1.iter().zip(c2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(diff);
    }

    #[test]
    fn test_freq_scaling_factor_doubles_angle() {
        let dim = 8;
        let half = dim / 2;
        let (cos1, sin1) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let (cos2, sin2) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 2.0);
        let off1_p2 = 2 * half;
        let off2_p1 = 1 * half;
        for i in 0..half {
            assert!((cos1[off1_p2 + i] - cos2[off2_p1 + i]).abs() < 1e-5);
            assert!((sin1[off1_p2 + i] - sin2[off2_p1 + i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_freq_all_finite() {
        let (cos, sin) = scalar_precompute_freqs(128, 512, DEFAULT_BASE, 1.0);
        assert!(cos.iter().all(|x| x.is_finite()));
        assert!(sin.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_freq_bounded() {
        let (cos, sin) = scalar_precompute_freqs(64, 256, DEFAULT_BASE, 1.0);
        assert!(cos.iter().all(|x| x.abs() <= 1.0 + EPS));
        assert!(sin.iter().all(|x| x.abs() <= 1.0 + EPS));
    }

    #[test]
    fn test_freq_dim4_manual() {
        let (cos, sin) = scalar_precompute_freqs(4, 2, DEFAULT_BASE, 1.0);
        assert!((cos[0] - 1.0).abs() < EPS);
        assert!((cos[1] - 1.0).abs() < EPS);
        assert!(sin[0].abs() < EPS);
        assert!(sin[1].abs() < EPS);
    }

    #[test]
    fn test_freq_higher_dims_rotate_slower() {
        let dim = 16;
        let half = dim / 2;
        let (_, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let sin_pair0 = sin[half].abs();
        let sin_pair_last = sin[half + half - 1].abs();
        assert!(sin_pair0 > sin_pair_last, "lower dims rotate faster");
    }

    // -- Edge cases (10+) ----------------------------------------------------

    #[test]
    fn test_edge_dim2() {
        let dim = 2;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = vec![1.0, 0.0];
        scalar_rope_apply(&mut data, &cos, &sin, dim, 1);
        let norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
        assert!((norm - 1.0).abs() < EPS);
    }

    #[test]
    fn test_edge_dim2_neox() {
        let dim = 2;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = vec![1.0, 0.0];
        scalar_rope_neox(&mut data, &cos, &sin, dim, 1);
        let norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
        assert!((norm - 1.0).abs() < EPS);
    }

    #[test]
    fn test_edge_zero_vector() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = vec![0.0f32; dim];
        scalar_rope_apply(&mut data, &cos, &sin, dim, 2);
        assert!(data.iter().all(|x| x.abs() < EPS));
    }

    #[test]
    fn test_edge_zero_vector_neox() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = vec![0.0f32; dim];
        scalar_rope_neox(&mut data, &cos, &sin, dim, 2);
        assert!(data.iter().all(|x| x.abs() < EPS));
    }

    #[test]
    fn test_edge_single_position_table() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 1, DEFAULT_BASE, 1.0);
        assert_eq!(cos.len(), dim / 2);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply(&mut data, &cos, &sin, dim, 0);
        assert_close(&data, &orig, EPS);
    }

    #[test]
    fn test_edge_large_position() {
        let dim = 8;
        let pos = 10_000;
        let (cos, sin) = scalar_precompute_freqs(dim, pos + 1, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        scalar_rope_apply(&mut data, &cos, &sin, dim, pos);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_edge_very_small_base() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, 1.0, 1.0);
        let mut data = make_data(dim);
        scalar_rope_apply(&mut data, &cos, &sin, dim, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_edge_very_large_base() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, 1e12, 1.0);
        let mut data = make_data(dim);
        scalar_rope_apply(&mut data, &cos, &sin, dim, 1);
        assert!(data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_edge_dim4_half_rotary() {
        let dim = 4;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_half(&mut data, &cos, &sin, dim, 1);
        assert_close(&data[2..], &orig[2..], EPS);
    }

    #[test]
    fn test_edge_negative_values() {
        let dim = 8;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data: Vec<f32> = (0..dim).map(|i| -(i as f32) - 1.0).collect();
        let orig_norm: f32 = data.iter().map(|x| x * x).sum();
        scalar_rope_apply(&mut data, &cos, &sin, dim, 2);
        let new_norm: f32 = data.iter().map(|x| x * x).sum();
        assert!((orig_norm - new_norm).abs() < 1e-3);
    }

    // -- Cache consistency (5+) ----------------------------------------------

    #[test]
    fn test_cache_matches_fresh_apply() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
        let mut d1 = make_data(dim);
        let mut d2 = d1.clone();
        scalar_rope_apply(&mut d1, &cos, &sin, dim, 5);
        scalar_rope_apply_with_cache(&mut d2, &cos, &sin, dim, 5);
        assert_close(&d1, &d2, EPS);
    }

    #[test]
    fn test_cache_multiple_positions() {
        let dim = 16;
        let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
        for pos in 0..16 {
            let mut d1 = make_data(dim);
            let mut d2 = d1.clone();
            scalar_rope_apply(&mut d1, &cos, &sin, dim, pos);
            scalar_rope_apply_with_cache(&mut d2, &cos, &sin, dim, pos);
            assert_close(&d1, &d2, EPS);
        }
    }

    #[test]
    fn test_cache_reuse_same_tables() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut d1 = make_data(dim);
        let mut d2 = make_data(dim);
        scalar_rope_apply_with_cache(&mut d1, &cos, &sin, dim, 1);
        scalar_rope_apply_with_cache(&mut d2, &cos, &sin, dim, 2);
        let differs = d1.iter().zip(d2.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(differs);
    }

    #[test]
    fn test_cache_identity_at_pos0() {
        let dim = 32;
        let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
        let mut data = make_data(dim);
        let orig = data.clone();
        scalar_rope_apply_with_cache(&mut data, &cos, &sin, dim, 0);
        assert_close(&data, &orig, EPS);
    }

    #[test]
    fn test_cache_norm_preservation() {
        let dim = 64;
        let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
        let data = make_data_seeded(dim, 101);
        let n0: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        scalar_rope_apply_with_cache(&mut rotated, &cos, &sin, dim, 10);
        let n1: f32 = rotated.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-3);
    }

    // -- NEON vs scalar parity (aarch64 only) --------------------------------

    #[cfg(target_arch = "aarch64")]
    mod neon_parity {
        use super::*;

        #[test]
        fn test_neon_precompute_matches_scalar() {
            let dim = 64;
            let max_seq = 32;
            let (sc, ss) = scalar_precompute_freqs(dim, max_seq, DEFAULT_BASE, 1.0);
            let (nc, ns) = unsafe { neon_rope_precompute_freqs(dim, max_seq, DEFAULT_BASE, 1.0) };
            assert_close(&sc, &nc, EPS);
            assert_close(&ss, &ns, EPS);
        }

        #[test]
        fn test_neon_apply_matches_scalar_dim8() {
            let dim = 8;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut neon_data = make_data(dim);
            let mut scalar_data = neon_data.clone();
            unsafe { neon_rope_apply(&mut neon_data, &cos, &sin, dim, 2) };
            scalar_rope_apply(&mut scalar_data, &cos, &sin, dim, 2);
            assert_close(&neon_data, &scalar_data, EPS);
        }

        #[test]
        fn test_neon_apply_matches_scalar_dim32() {
            let dim = 32;
            let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
            for pos in 0..8 {
                let mut nd = make_data_seeded(dim, pos as u32 + 1);
                let mut sd = nd.clone();
                unsafe { neon_rope_apply(&mut nd, &cos, &sin, dim, pos) };
                scalar_rope_apply(&mut sd, &cos, &sin, dim, pos);
                assert_close(&nd, &sd, EPS);
            }
        }

        #[test]
        fn test_neon_apply_matches_scalar_dim64() {
            let dim = 64;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 99);
            let mut sd = nd.clone();
            unsafe { neon_rope_apply(&mut nd, &cos, &sin, dim, 3) };
            scalar_rope_apply(&mut sd, &cos, &sin, dim, 3);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_apply_matches_scalar_dim128() {
            let dim = 128;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 200);
            let mut sd = nd.clone();
            unsafe { neon_rope_apply(&mut nd, &cos, &sin, dim, 2) };
            scalar_rope_apply(&mut sd, &cos, &sin, dim, 2);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_apply_matches_scalar_dim256() {
            let dim = 256;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 300);
            let mut sd = nd.clone();
            unsafe { neon_rope_apply(&mut nd, &cos, &sin, dim, 1) };
            scalar_rope_apply(&mut sd, &cos, &sin, dim, 1);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_half_matches_scalar_dim32() {
            let dim = 32;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data(dim);
            let mut sd = nd.clone();
            unsafe { neon_rope_apply_half(&mut nd, &cos, &sin, dim, 2) };
            scalar_rope_apply_half(&mut sd, &cos, &sin, dim, 2);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_half_matches_scalar_dim64() {
            let dim = 64;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 55);
            let mut sd = nd.clone();
            unsafe { neon_rope_apply_half(&mut nd, &cos, &sin, dim, 1) };
            scalar_rope_apply_half(&mut sd, &cos, &sin, dim, 1);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_half_matches_scalar_dim128() {
            let dim = 128;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 88);
            let mut sd = nd.clone();
            unsafe { neon_rope_apply_half(&mut nd, &cos, &sin, dim, 3) };
            scalar_rope_apply_half(&mut sd, &cos, &sin, dim, 3);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_neox_matches_scalar_dim16() {
            let dim = 16;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data(dim);
            let mut sd = nd.clone();
            unsafe { neon_rope_neox(&mut nd, &cos, &sin, dim, 2) };
            scalar_rope_neox(&mut sd, &cos, &sin, dim, 2);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_neox_matches_scalar_dim32() {
            let dim = 32;
            let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
            for pos in 0..8 {
                let mut nd = make_data_seeded(dim, pos as u32 + 10);
                let mut sd = nd.clone();
                unsafe { neon_rope_neox(&mut nd, &cos, &sin, dim, pos) };
                scalar_rope_neox(&mut sd, &cos, &sin, dim, pos);
                assert_close(&nd, &sd, EPS);
            }
        }

        #[test]
        fn test_neon_neox_matches_scalar_dim64() {
            let dim = 64;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 77);
            let mut sd = nd.clone();
            unsafe { neon_rope_neox(&mut nd, &cos, &sin, dim, 3) };
            scalar_rope_neox(&mut sd, &cos, &sin, dim, 3);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_neox_matches_scalar_dim128() {
            let dim = 128;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim, 150);
            let mut sd = nd.clone();
            unsafe { neon_rope_neox(&mut nd, &cos, &sin, dim, 2) };
            scalar_rope_neox(&mut sd, &cos, &sin, dim, 2);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_batched_matches_scalar() {
            let dim = 16;
            let num_heads = 4;
            let seq = 3;
            let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
            let mut nd = make_data(dim * num_heads * seq);
            let mut sd = nd.clone();
            unsafe { neon_rope_batched(&mut nd, &cos, &sin, dim, num_heads, 1, seq) };
            scalar_rope_batched(&mut sd, &cos, &sin, dim, num_heads, 1, seq);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_batched_matches_scalar_dim64() {
            let dim = 64;
            let num_heads = 8;
            let seq = 4;
            let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
            let mut nd = make_data_seeded(dim * num_heads * seq, 42);
            let mut sd = nd.clone();
            unsafe { neon_rope_batched(&mut nd, &cos, &sin, dim, num_heads, 0, seq) };
            scalar_rope_batched(&mut sd, &cos, &sin, dim, num_heads, 0, seq);
            assert_close(&nd, &sd, EPS);
        }

        #[test]
        fn test_neon_cache_matches_scalar() {
            let dim = 32;
            let (cos, sin) = scalar_precompute_freqs(dim, 8, DEFAULT_BASE, 1.0);
            for pos in 0..8 {
                let mut nd = make_data_seeded(dim, pos as u32 + 50);
                let mut sd = nd.clone();
                unsafe { neon_rope_apply_with_cache(&mut nd, &cos, &sin, dim, pos) };
                scalar_rope_apply_with_cache(&mut sd, &cos, &sin, dim, pos);
                assert_close(&nd, &sd, EPS);
            }
        }

        #[test]
        fn test_neon_apply_identity_pos0() {
            let dim = 64;
            let (cos, sin) = scalar_precompute_freqs(dim, 2, DEFAULT_BASE, 1.0);
            let mut data = make_data(dim);
            let orig = data.clone();
            unsafe { neon_rope_apply(&mut data, &cos, &sin, dim, 0) };
            assert_close(&data, &orig, EPS);
        }

        #[test]
        fn test_neon_apply_norm_preservation() {
            let dim = 128;
            let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
            let data = make_data_seeded(dim, 555);
            let n0: f32 = data.iter().map(|x| x * x).sum();
            let mut rotated = data;
            unsafe { neon_rope_apply(&mut rotated, &cos, &sin, dim, 10) };
            let n1: f32 = rotated.iter().map(|x| x * x).sum();
            assert!((n0 - n1).abs() < 1e-3);
        }

        #[test]
        fn test_neon_neox_norm_preservation() {
            let dim = 64;
            let (cos, sin) = scalar_precompute_freqs(dim, 16, DEFAULT_BASE, 1.0);
            let data = make_data_seeded(dim, 666);
            let n0: f32 = data.iter().map(|x| x * x).sum();
            let mut rotated = data;
            unsafe { neon_rope_neox(&mut rotated, &cos, &sin, dim, 7) };
            let n1: f32 = rotated.iter().map(|x| x * x).sum();
            assert!((n0 - n1).abs() < 1e-3);
        }

        #[test]
        fn test_neon_half_second_half_unchanged() {
            let dim = 64;
            let (cos, sin) = scalar_precompute_freqs(dim, 4, DEFAULT_BASE, 1.0);
            let mut data = make_data(dim);
            let orig = data.clone();
            unsafe { neon_rope_apply_half(&mut data, &cos, &sin, dim, 2) };
            assert_close(&data[32..], &orig[32..], EPS);
        }
    }
}
