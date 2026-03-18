//! NEON-optimized RoPE v4 (Rotary Position Embedding) for AArch64.
//!
//! Provides:
//! - Direct Q/K application without precomputed tables
//! - Cos/sin cache precomputation
//! - Cached single-head application
//! - Batched multi-head application
//! - NeoX-style (half-split) layout
//!
//! All public functions are gated on `target_arch = "aarch64"`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar helpers (used as reference and fallback) ─────────────────

/// Scalar RoPE rotation for one pair at dimension index `i`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn scalar_rope_pair(x0: f32, x1: f32, cos_val: f32, sin_val: f32) -> (f32, f32) {
    (x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val)
}

/// Compute theta for dimension pair index `i` at `position`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn theta(position: usize, i: usize, head_dim: usize, base: f32) -> f32 {
    let exponent = -(2.0 * i as f32) / head_dim as f32;
    position as f32 * base.powf(exponent)
}

// ── Public API ──────────────────────────────────────────────────────

/// Apply RoPE to Q and K vectors in-place (no precomputed tables).
///
/// Both `q` and `k` must have length ≥ `head_dim`. Only the first
/// `head_dim` elements are rotated.
///
/// # Panics
///
/// Panics if `head_dim` is odd, zero, or if slices are too short.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `q` and `k` must have length ≥ `head_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply_f32(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    position: usize,
    base: f32,
) {
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "head_dim must be even and non-zero");
    assert!(q.len() >= head_dim, "q too short");
    assert!(k.len() >= head_dim, "k too short");

    let half_dim = head_dim / 2;

    // Process 2 pairs (4 floats) at a time with NEON.
    let neon_chunks = half_dim / 2;
    let sign_mask = unsafe { vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr()) };

    for c in 0..neon_chunks {
        let i0 = c * 2;
        let i1 = i0 + 1;
        let data_idx = c * 4;

        let angle0 = theta(position, i0, head_dim, base);
        let angle1 = theta(position, i1, head_dim, base);

        let (c0, s0) = (angle0.cos(), angle0.sin());
        let (c1, s1) = (angle1.cos(), angle1.sin());

        unsafe {
            let cos_v = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sin_v = vld1q_f32([s0, s0, s1, s1].as_ptr());

            // Q
            let qv = vld1q_f32(q.as_ptr().add(data_idx));
            let qs = vrev64q_f32(qv);
            let qr = vaddq_f32(vmulq_f32(qv, cos_v), vmulq_f32(vmulq_f32(qs, sign_mask), sin_v));
            vst1q_f32(q.as_mut_ptr().add(data_idx), qr);

            // K
            let kv = vld1q_f32(k.as_ptr().add(data_idx));
            let ks = vrev64q_f32(kv);
            let kr = vaddq_f32(vmulq_f32(kv, cos_v), vmulq_f32(vmulq_f32(ks, sign_mask), sin_v));
            vst1q_f32(k.as_mut_ptr().add(data_idx), kr);
        }
    }

    // Scalar tail
    let processed_pairs = neon_chunks * 2;
    for i in processed_pairs..half_dim {
        let idx = i * 2;
        let angle = theta(position, i, head_dim, base);
        let (cv, sv) = (angle.cos(), angle.sin());

        let (q0, q1) = scalar_rope_pair(q[idx], q[idx + 1], cv, sv);
        q[idx] = q0;
        q[idx + 1] = q1;

        let (k0, k1) = scalar_rope_pair(k[idx], k[idx + 1], cv, sv);
        k[idx] = k0;
        k[idx + 1] = k1;
    }
}

/// Precompute cos/sin cache for RoPE.
///
/// Returns `(cos_cache, sin_cache)` where each has layout
/// `cache[pos * half_dim + i]` for position `pos` and pair index `i`,
/// with `half_dim = head_dim / 2`.
///
/// # Panics
///
/// Panics if `head_dim` is odd or zero.
#[cfg(target_arch = "aarch64")]
pub fn neon_rope_build_cos_sin_cache(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "head_dim must be even and non-zero");

    let half_dim = head_dim / 2;
    let total = max_seq_len * half_dim;
    let mut cos_cache = Vec::with_capacity(total);
    let mut sin_cache = Vec::with_capacity(total);

    for pos in 0..max_seq_len {
        for i in 0..half_dim {
            let angle = theta(pos, i, head_dim, base);
            cos_cache.push(angle.cos());
            sin_cache.push(angle.sin());
        }
    }

    (cos_cache, sin_cache)
}

/// Apply RoPE in-place using precomputed cos/sin cache.
///
/// `data` must have length ≥ `head_dim`. The cache must cover `position`.
///
/// # Panics
///
/// Panics if `head_dim` is odd/zero, data is too short, or cache doesn't
/// cover the requested position.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `data` must have length ≥ `head_dim` and caches must cover `position`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply_cached_f32(
    data: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    position: usize,
    head_dim: usize,
) {
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "head_dim must be even and non-zero");
    assert!(data.len() >= head_dim, "data too short");

    let half_dim = head_dim / 2;
    let table_offset = position * half_dim;
    assert!(
        cos_cache.len() >= table_offset + half_dim,
        "cos_cache too short for position {position}"
    );
    assert!(
        sin_cache.len() >= table_offset + half_dim,
        "sin_cache too short for position {position}"
    );

    let neon_chunks = half_dim / 2;
    let sign_mask = unsafe { vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr()) };

    for c in 0..neon_chunks {
        let data_idx = c * 4;
        let ti = table_offset + c * 2;

        unsafe {
            let c0 = *cos_cache.get_unchecked(ti);
            let c1 = *cos_cache.get_unchecked(ti + 1);
            let s0 = *sin_cache.get_unchecked(ti);
            let s1 = *sin_cache.get_unchecked(ti + 1);

            let cos_v = vld1q_f32([c0, c0, c1, c1].as_ptr());
            let sin_v = vld1q_f32([s0, s0, s1, s1].as_ptr());

            let vals = vld1q_f32(data.as_ptr().add(data_idx));
            let swapped = vrev64q_f32(vals);
            let rotated =
                vaddq_f32(vmulq_f32(vals, cos_v), vmulq_f32(vmulq_f32(swapped, sign_mask), sin_v));
            vst1q_f32(data.as_mut_ptr().add(data_idx), rotated);
        }
    }

    // Scalar tail
    let processed = neon_chunks * 2;
    for i in processed..half_dim {
        let idx = i * 2;
        let cv = cos_cache[table_offset + i];
        let sv = sin_cache[table_offset + i];
        let (r0, r1) = scalar_rope_pair(data[idx], data[idx + 1], cv, sv);
        data[idx] = r0;
        data[idx + 1] = r1;
    }
}

/// Batched RoPE: apply to `[batch × num_heads × seq_len × head_dim]` data.
///
/// Data layout is row-major: the innermost dimension is `head_dim`, then
/// `seq_len`, then `num_heads`, then `batch`.
///
/// The cache must cover at least `seq_len` positions.
///
/// # Panics
///
/// Panics if `data.len() < batch * num_heads * seq_len * head_dim` or
/// caches are too small.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `data` must have length ≥ `batch * num_heads * seq_len * head_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply_batch_f32(
    data: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let total = batch * num_heads * seq_len * head_dim;
    assert!(data.len() >= total, "data too short for batch RoPE");

    for b in 0..batch {
        for h in 0..num_heads {
            for s in 0..seq_len {
                let offset = ((b * num_heads + h) * seq_len + s) * head_dim;
                unsafe {
                    neon_rope_apply_cached_f32(
                        &mut data[offset..offset + head_dim],
                        cos_cache,
                        sin_cache,
                        s,
                        head_dim,
                    );
                }
            }
        }
    }
}

/// NeoX-style RoPE: half-split layout.
///
/// Instead of interleaved pairs `(x0,x1), (x2,x3), ...`, NeoX uses
/// `(x[i], x[i + half_dim])` as rotation pairs:
///
/// ```text
/// x_i'            = x_i            * cos(theta) - x_{i+half_dim} * sin(theta)
/// x_{i+half_dim}' = x_i            * sin(theta) + x_{i+half_dim} * cos(theta)
/// ```
///
/// # Panics
///
/// Panics if `head_dim` is odd/zero, data is too short, or cache doesn't
/// cover the requested position.
///
/// # Safety
///
/// Caller must ensure the target supports NEON instructions (aarch64).
/// `data` must have length ≥ `head_dim` and caches must cover `position`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_apply_neox_f32(
    data: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    position: usize,
    head_dim: usize,
) {
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "head_dim must be even and non-zero");
    assert!(data.len() >= head_dim, "data too short");

    let half_dim = head_dim / 2;
    let table_offset = position * half_dim;
    assert!(
        cos_cache.len() >= table_offset + half_dim,
        "cos_cache too short for position {position}"
    );
    assert!(
        sin_cache.len() >= table_offset + half_dim,
        "sin_cache too short for position {position}"
    );

    // NEON: process 4 pairs at a time
    let neon_chunks = half_dim / 4;
    for c in 0..neon_chunks {
        let i_base = c * 4;
        let ti = table_offset + i_base;

        unsafe {
            let cos_v = vld1q_f32(cos_cache.as_ptr().add(ti));
            let sin_v = vld1q_f32(sin_cache.as_ptr().add(ti));

            let lo = vld1q_f32(data.as_ptr().add(i_base));
            let hi = vld1q_f32(data.as_ptr().add(i_base + half_dim));

            let new_lo = vsubq_f32(vmulq_f32(lo, cos_v), vmulq_f32(hi, sin_v));
            let new_hi = vaddq_f32(vmulq_f32(lo, sin_v), vmulq_f32(hi, cos_v));

            vst1q_f32(data.as_mut_ptr().add(i_base), new_lo);
            vst1q_f32(data.as_mut_ptr().add(i_base + half_dim), new_hi);
        }
    }

    // Scalar tail
    let processed = neon_chunks * 4;
    for i in processed..half_dim {
        let cv = cos_cache[table_offset + i];
        let sv = sin_cache[table_offset + i];
        let lo = data[i];
        let hi = data[i + half_dim];
        data[i] = lo * cv - hi * sv;
        data[i + half_dim] = lo * sv + hi * cv;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const BASE: f32 = 10_000.0;
    const EPS: f32 = 1e-5;

    // ── Scalar reference implementation ─────────────────────────────

    fn scalar_rope_apply(data: &mut [f32], head_dim: usize, position: usize, base: f32) {
        let half_dim = head_dim / 2;
        for i in 0..half_dim {
            let angle = theta(position, i, head_dim, base);
            let (c, s) = (angle.cos(), angle.sin());
            let idx = i * 2;
            let x0 = data[idx];
            let x1 = data[idx + 1];
            data[idx] = x0 * c - x1 * s;
            data[idx + 1] = x0 * s + x1 * c;
        }
    }

    fn scalar_rope_apply_neox(
        data: &mut [f32],
        cos_cache: &[f32],
        sin_cache: &[f32],
        position: usize,
        head_dim: usize,
    ) {
        let half_dim = head_dim / 2;
        let off = position * half_dim;
        for i in 0..half_dim {
            let cv = cos_cache[off + i];
            let sv = sin_cache[off + i];
            let lo = data[i];
            let hi = data[i + half_dim];
            data[i] = lo * cv - hi * sv;
            data[i + half_dim] = lo * sv + hi * cv;
        }
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vecs_approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    fn make_data(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect()
    }

    // ── neon_rope_apply_f32 correctness ─────────────────────────────

    #[test]
    fn test_apply_f32_dim2_pos0() {
        let mut q = make_data(2);
        let mut k = make_data(2);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 2, 0, BASE) };
        scalar_rope_apply(&mut q_ref, 2, 0, BASE);
        scalar_rope_apply(&mut k_ref, 2, 0, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_dim2_pos1() {
        let mut q = make_data(2);
        let mut k = make_data(2);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 2, 1, BASE) };
        scalar_rope_apply(&mut q_ref, 2, 1, BASE);
        scalar_rope_apply(&mut k_ref, 2, 1, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_dim32() {
        let mut q = make_data(32);
        let mut k = make_data(32);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 32, 5, BASE) };
        scalar_rope_apply(&mut q_ref, 32, 5, BASE);
        scalar_rope_apply(&mut k_ref, 32, 5, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_dim64() {
        let mut q = make_data(64);
        let mut k = make_data(64);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 64, 10, BASE) };
        scalar_rope_apply(&mut q_ref, 64, 10, BASE);
        scalar_rope_apply(&mut k_ref, 64, 10, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_dim128() {
        let mut q = make_data(128);
        let mut k = make_data(128);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 128, 42, BASE) };
        scalar_rope_apply(&mut q_ref, 128, 42, BASE);
        scalar_rope_apply(&mut k_ref, 128, 42, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_dim256() {
        let mut q = make_data(256);
        let mut k = make_data(256);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 256, 7, BASE) };
        scalar_rope_apply(&mut q_ref, 256, 7, BASE);
        scalar_rope_apply(&mut k_ref, 256, 7, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_pos0_identity_like() {
        // At position 0, theta = 0, cos=1, sin=0 → output == input
        let mut q = make_data(64);
        let mut k = make_data(64);
        let q_orig = q.clone();
        let k_orig = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 64, 0, BASE) };
        assert!(vecs_approx_eq(&q, &q_orig, EPS));
        assert!(vecs_approx_eq(&k, &k_orig, EPS));
    }

    #[test]
    fn test_apply_f32_large_position() {
        let pos = 100_000;
        let mut q = make_data(64);
        let mut k = make_data(64);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 64, pos, BASE) };
        scalar_rope_apply(&mut q_ref, 64, pos, BASE);
        scalar_rope_apply(&mut k_ref, 64, pos, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_different_base() {
        let base = 500.0;
        let mut q = make_data(32);
        let mut k = make_data(32);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 32, 3, base) };
        scalar_rope_apply(&mut q_ref, 32, 3, base);
        scalar_rope_apply(&mut k_ref, 32, 3, base);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_q_k_independent() {
        // Changing K doesn't affect Q result
        let mut q1 = make_data(32);
        let mut k1 = make_data(32);
        let mut q2 = q1.clone();
        let mut k2 = [99.0; 32];
        unsafe { neon_rope_apply_f32(&mut q1, &mut k1, 32, 5, BASE) };
        unsafe { neon_rope_apply_f32(&mut q2, &mut k2, 32, 5, BASE) };
        assert!(vecs_approx_eq(&q1, &q2, EPS));
    }

    #[test]
    fn test_apply_f32_dim6_scalar_tail() {
        // head_dim=6: 1 NEON chunk (4 floats) + 1 scalar pair
        let mut q = make_data(6);
        let mut k = make_data(6);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 6, 3, BASE) };
        scalar_rope_apply(&mut q_ref, 6, 3, BASE);
        scalar_rope_apply(&mut k_ref, 6, 3, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_apply_f32_dim10() {
        // head_dim=10: 2 NEON chunks + 1 scalar pair
        let mut q = make_data(10);
        let mut k = make_data(10);
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 10, 2, BASE) };
        scalar_rope_apply(&mut q_ref, 10, 2, BASE);
        scalar_rope_apply(&mut k_ref, 10, 2, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    // ── Cache building ──────────────────────────────────────────────

    #[test]
    fn test_cache_length() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(64, 128, BASE);
        assert_eq!(cos.len(), 128 * 32);
        assert_eq!(sin.len(), 128 * 32);
    }

    #[test]
    fn test_cache_cos2_sin2_equals_1() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(64, 512, BASE);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(approx_eq(sum, 1.0, EPS), "cos²+sin²≠1 at index {i}: {sum}");
        }
    }

    #[test]
    fn test_cache_pos0_cos_is_one() {
        let (cos, _sin) = neon_rope_build_cos_sin_cache(64, 8, BASE);
        let half = 32;
        for i in 0..half {
            assert!(approx_eq(cos[i], 1.0, EPS), "cos[{i}] at pos 0 should be 1.0");
        }
    }

    #[test]
    fn test_cache_pos0_sin_is_zero() {
        let (_cos, sin) = neon_rope_build_cos_sin_cache(64, 8, BASE);
        let half = 32;
        for i in 0..half {
            assert!(approx_eq(sin[i], 0.0, EPS), "sin[{i}] at pos 0 should be 0.0");
        }
    }

    #[test]
    fn test_cache_dim32() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(32, 16, BASE);
        assert_eq!(cos.len(), 16 * 16);
        assert_eq!(sin.len(), 16 * 16);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(approx_eq(sum, 1.0, EPS));
        }
    }

    #[test]
    fn test_cache_dim128() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(128, 64, BASE);
        assert_eq!(cos.len(), 64 * 64);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(approx_eq(sum, 1.0, EPS));
        }
    }

    #[test]
    fn test_cache_dim256() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(256, 32, BASE);
        assert_eq!(cos.len(), 32 * 128);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(approx_eq(sum, 1.0, EPS));
        }
    }

    #[test]
    fn test_cache_small_base() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(32, 8, 100.0);
        for i in 0..cos.len() {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(approx_eq(sum, 1.0, EPS));
        }
    }

    #[test]
    fn test_cache_single_position() {
        let (cos, sin) = neon_rope_build_cos_sin_cache(64, 1, BASE);
        assert_eq!(cos.len(), 32);
        // pos 0 → all cos=1, sin=0
        for i in 0..32 {
            assert!(approx_eq(cos[i], 1.0, EPS));
            assert!(approx_eq(sin[i], 0.0, EPS));
        }
    }

    #[test]
    fn test_cache_matches_direct_theta() {
        let head_dim = 64;
        let max_seq = 16;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, max_seq, BASE);
        let half = head_dim / 2;
        for pos in 0..max_seq {
            for i in 0..half {
                let angle = theta(pos, i, head_dim, BASE);
                let idx = pos * half + i;
                assert!(approx_eq(cos[idx], angle.cos(), EPS));
                assert!(approx_eq(sin[idx], angle.sin(), EPS));
            }
        }
    }

    // ── Cached application ──────────────────────────────────────────

    #[test]
    fn test_cached_matches_direct_dim32() {
        let head_dim = 32;
        let pos = 5;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_cached_matches_direct_dim64() {
        let head_dim = 64;
        let pos = 10;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_cached_matches_direct_dim128() {
        let head_dim = 128;
        let pos = 3;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_cached_matches_direct_dim256() {
        let head_dim = 256;
        let pos = 1;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_cached_pos0_identity() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let mut data = make_data(head_dim);
        let orig = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, 0, head_dim) };
        assert!(vecs_approx_eq(&data, &orig, EPS));
    }

    #[test]
    fn test_cached_dim2() {
        let head_dim = 2;
        let pos = 7;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_cached_dim6_tail() {
        let head_dim = 6;
        let pos = 2;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_cached_vs_direct_apply_f32() {
        // Verify cached path matches the direct (no-cache) path
        let head_dim = 64;
        let pos = 11;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut cached = make_data(head_dim);
        let mut direct_q = cached.clone();
        let mut direct_k = make_data(head_dim);

        unsafe { neon_rope_apply_cached_f32(&mut cached, &cos, &sin, pos, head_dim) };
        unsafe { neon_rope_apply_f32(&mut direct_q, &mut direct_k, head_dim, pos, BASE) };
        assert!(vecs_approx_eq(&cached, &direct_q, EPS));
    }

    // ── Batch application ───────────────────────────────────────────

    #[test]
    fn test_batch_single_matches_cached() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let mut batch_data = make_data(head_dim);
        let mut single_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(&mut batch_data, &cos, &sin, 1, 1, 1, head_dim);
            neon_rope_apply_cached_f32(&mut single_data, &cos, &sin, 0, head_dim);
        }
        assert!(vecs_approx_eq(&batch_data, &single_data, EPS));
    }

    #[test]
    fn test_batch_multi_head_matches_sequential() {
        let head_dim = 32;
        let num_heads = 4;
        let seq_len = 1;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);

        let mut batch_data = make_data(num_heads * seq_len * head_dim);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(&mut batch_data, &cos, &sin, 1, num_heads, seq_len, head_dim);
        }
        // Apply sequentially
        for h in 0..num_heads {
            let off = h * head_dim;
            unsafe {
                neon_rope_apply_cached_f32(
                    &mut seq_data[off..off + head_dim],
                    &cos,
                    &sin,
                    0,
                    head_dim,
                );
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    #[test]
    fn test_batch_multi_seq_matches_sequential() {
        let head_dim = 32;
        let num_heads = 2;
        let seq_len = 4;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, seq_len, BASE);

        let total = num_heads * seq_len * head_dim;
        let mut batch_data = make_data(total);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(&mut batch_data, &cos, &sin, 1, num_heads, seq_len, head_dim);
        }
        for h in 0..num_heads {
            for s in 0..seq_len {
                let off = (h * seq_len + s) * head_dim;
                unsafe {
                    neon_rope_apply_cached_f32(
                        &mut seq_data[off..off + head_dim],
                        &cos,
                        &sin,
                        s,
                        head_dim,
                    );
                }
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    #[test]
    fn test_batch_2_matches_sequential() {
        let head_dim = 64;
        let batch = 2;
        let num_heads = 2;
        let seq_len = 3;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, seq_len, BASE);

        let total = batch * num_heads * seq_len * head_dim;
        let mut batch_data = make_data(total);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(
                &mut batch_data,
                &cos,
                &sin,
                batch,
                num_heads,
                seq_len,
                head_dim,
            );
        }
        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let off = ((b * num_heads + h) * seq_len + s) * head_dim;
                    unsafe {
                        neon_rope_apply_cached_f32(
                            &mut seq_data[off..off + head_dim],
                            &cos,
                            &sin,
                            s,
                            head_dim,
                        );
                    }
                }
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    #[test]
    fn test_batch_dim128() {
        let head_dim = 128;
        let batch = 1;
        let num_heads = 2;
        let seq_len = 2;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, seq_len, BASE);

        let total = batch * num_heads * seq_len * head_dim;
        let mut batch_data = make_data(total);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(
                &mut batch_data,
                &cos,
                &sin,
                batch,
                num_heads,
                seq_len,
                head_dim,
            );
        }
        for h in 0..num_heads {
            for s in 0..seq_len {
                let off = (h * seq_len + s) * head_dim;
                unsafe {
                    neon_rope_apply_cached_f32(
                        &mut seq_data[off..off + head_dim],
                        &cos,
                        &sin,
                        s,
                        head_dim,
                    );
                }
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    #[test]
    fn test_batch_dim256() {
        let head_dim = 256;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 2, BASE);
        let total = 1 * 1 * 2 * head_dim;
        let mut batch_data = make_data(total);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(&mut batch_data, &cos, &sin, 1, 1, 2, head_dim);
        }
        for s in 0..2 {
            let off = s * head_dim;
            unsafe {
                neon_rope_apply_cached_f32(
                    &mut seq_data[off..off + head_dim],
                    &cos,
                    &sin,
                    s,
                    head_dim,
                );
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    // ── NeoX layout ─────────────────────────────────────────────────

    #[test]
    fn test_neox_matches_scalar_dim32() {
        let head_dim = 32;
        let pos = 5;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_neox_matches_scalar_dim64() {
        let head_dim = 64;
        let pos = 10;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_neox_matches_scalar_dim128() {
        let head_dim = 128;
        let pos = 3;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_neox_matches_scalar_dim256() {
        let head_dim = 256;
        let pos = 1;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_neox_pos0_identity() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let mut data = make_data(head_dim);
        let orig = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, 0, head_dim) };
        assert!(vecs_approx_eq(&data, &orig, EPS));
    }

    #[test]
    fn test_neox_dim2() {
        // NeoX with dim=2: pair is (data[0], data[1]) same as standard
        let head_dim = 2;
        let pos = 4;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_neox_dim6_tail() {
        let head_dim = 6;
        let pos = 2;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_neox_differs_from_standard() {
        // NeoX and standard layouts should produce different results (unless dim=2)
        let head_dim = 64;
        let pos = 5;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut neox = make_data(head_dim);
        let mut standard = neox.clone();
        unsafe { neon_rope_apply_neox_f32(&mut neox, &cos, &sin, pos, head_dim) };
        unsafe { neon_rope_apply_cached_f32(&mut standard, &cos, &sin, pos, head_dim) };
        // They should NOT be equal for dim>2
        assert!(!vecs_approx_eq(&neox, &standard, EPS));
    }

    #[test]
    fn test_neox_large_position() {
        let head_dim = 64;
        let pos = 50_000;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, pos + 1, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    // ── Rotation invariants ─────────────────────────────────────────

    #[test]
    fn test_rotation_preserves_norm_dim32() {
        let head_dim = 32;
        let data = make_data(head_dim);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 64, BASE);
        unsafe { neon_rope_apply_cached_f32(&mut rotated, &cos, &sin, 7, head_dim) };
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_before, norm_after, 1e-3));
    }

    #[test]
    fn test_rotation_preserves_norm_dim64() {
        let head_dim = 64;
        let data = make_data(head_dim);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 64, BASE);
        unsafe { neon_rope_apply_cached_f32(&mut rotated, &cos, &sin, 15, head_dim) };
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_before, norm_after, 1e-3));
    }

    #[test]
    fn test_rotation_preserves_norm_dim128() {
        let head_dim = 128;
        let data = make_data(head_dim);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 64, BASE);
        unsafe { neon_rope_apply_cached_f32(&mut rotated, &cos, &sin, 20, head_dim) };
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_before, norm_after, 1e-2));
    }

    #[test]
    fn test_rotation_preserves_norm_dim256() {
        let head_dim = 256;
        let data = make_data(head_dim);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 64, BASE);
        unsafe { neon_rope_apply_cached_f32(&mut rotated, &cos, &sin, 5, head_dim) };
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_before, norm_after, 1e-1));
    }

    #[test]
    fn test_neox_preserves_norm() {
        let head_dim = 64;
        let data = make_data(head_dim);
        let norm_before: f32 = data.iter().map(|x| x * x).sum();
        let mut rotated = data;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 64, BASE);
        unsafe { neon_rope_apply_neox_f32(&mut rotated, &cos, &sin, 10, head_dim) };
        let norm_after: f32 = rotated.iter().map(|x| x * x).sum();
        assert!(approx_eq(norm_before, norm_after, 1e-3));
    }

    // ── Determinism ─────────────────────────────────────────────────

    #[test]
    fn test_deterministic_apply() {
        let head_dim = 64;
        let pos = 7;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let orig = make_data(head_dim);

        let mut a = orig.clone();
        let mut b = orig.clone();
        unsafe { neon_rope_apply_cached_f32(&mut a, &cos, &sin, pos, head_dim) };
        unsafe { neon_rope_apply_cached_f32(&mut b, &cos, &sin, pos, head_dim) };
        assert_eq!(a, b); // Exact equality — same inputs, same NEON path
    }

    #[test]
    fn test_deterministic_neox() {
        let head_dim = 64;
        let pos = 3;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let orig = make_data(head_dim);

        let mut a = orig.clone();
        let mut b = orig.clone();
        unsafe { neon_rope_apply_neox_f32(&mut a, &cos, &sin, pos, head_dim) };
        unsafe { neon_rope_apply_neox_f32(&mut b, &cos, &sin, pos, head_dim) };
        assert_eq!(a, b);
    }

    #[test]
    fn test_deterministic_batch() {
        let head_dim = 32;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let orig = make_data(2 * 2 * 4 * head_dim);

        let mut a = orig.clone();
        let mut b = orig.clone();
        unsafe { neon_rope_apply_batch_f32(&mut a, &cos, &sin, 2, 2, 4, head_dim) };
        unsafe { neon_rope_apply_batch_f32(&mut b, &cos, &sin, 2, 2, 4, head_dim) };
        assert_eq!(a, b);
    }

    // ── Different positions produce different results ────────────────

    #[test]
    fn test_different_positions_differ_cached() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);

        let mut a = make_data(head_dim);
        let mut b = a.clone();
        unsafe { neon_rope_apply_cached_f32(&mut a, &cos, &sin, 1, head_dim) };
        unsafe { neon_rope_apply_cached_f32(&mut b, &cos, &sin, 2, head_dim) };
        assert!(!vecs_approx_eq(&a, &b, EPS));
    }

    #[test]
    fn test_different_positions_differ_direct() {
        let mut q1 = make_data(64);
        let mut k1 = make_data(64);
        let mut q2 = q1.clone();
        let mut k2 = k1.clone();
        unsafe { neon_rope_apply_f32(&mut q1, &mut k1, 64, 1, BASE) };
        unsafe { neon_rope_apply_f32(&mut q2, &mut k2, 64, 2, BASE) };
        assert!(!vecs_approx_eq(&q1, &q2, EPS));
    }

    #[test]
    fn test_different_positions_differ_neox() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut a = make_data(head_dim);
        let mut b = a.clone();
        unsafe { neon_rope_apply_neox_f32(&mut a, &cos, &sin, 3, head_dim) };
        unsafe { neon_rope_apply_neox_f32(&mut b, &cos, &sin, 7, head_dim) };
        assert!(!vecs_approx_eq(&a, &b, EPS));
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_zeros_stay_zero() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = vec![0.0f32; head_dim];
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, 5, head_dim) };
        assert!(data.iter().all(|&x| x.abs() < EPS));
    }

    #[test]
    fn test_zeros_stay_zero_neox() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);
        let mut data = vec![0.0f32; head_dim];
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, 5, head_dim) };
        assert!(data.iter().all(|&x| x.abs() < EPS));
    }

    #[test]
    fn test_zeros_stay_zero_direct() {
        let mut q = [0.0f32; 64];
        let mut k = [0.0f32; 64];
        unsafe { neon_rope_apply_f32(&mut q, &mut k, 64, 5, BASE) };
        assert!(q.iter().all(|&x| x.abs() < EPS));
        assert!(k.iter().all(|&x| x.abs() < EPS));
    }

    #[test]
    fn test_longer_data_only_head_dim_modified() {
        let head_dim = 32;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let mut data = make_data(64); // longer than head_dim
        let tail = data[32..].to_vec();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, 1, head_dim) };
        // Tail beyond head_dim should be unchanged
        assert_eq!(&data[32..], tail.as_slice());
    }

    #[test]
    fn test_longer_data_only_head_dim_modified_neox() {
        let head_dim = 32;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 4, BASE);
        let mut data = make_data(64);
        let tail = data[32..].to_vec();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, 1, head_dim) };
        assert_eq!(&data[32..], tail.as_slice());
    }

    // ── Multiple positions sweep ────────────────────────────────────

    #[test]
    fn test_sweep_positions_dim32() {
        let head_dim = 32;
        let max_pos = 20;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, max_pos, BASE);
        for pos in 0..max_pos {
            let mut data = make_data(head_dim);
            let mut data_ref = data.clone();
            unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
            scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
            assert!(vecs_approx_eq(&data, &data_ref, EPS), "mismatch at pos {pos}");
        }
    }

    #[test]
    fn test_sweep_positions_dim64() {
        let head_dim = 64;
        let max_pos = 20;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, max_pos, BASE);
        for pos in 0..max_pos {
            let mut data = make_data(head_dim);
            let mut data_ref = data.clone();
            unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
            scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
            assert!(vecs_approx_eq(&data, &data_ref, EPS), "mismatch at pos {pos}");
        }
    }

    #[test]
    fn test_sweep_positions_dim128() {
        let head_dim = 128;
        let max_pos = 16;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, max_pos, BASE);
        for pos in 0..max_pos {
            let mut data = make_data(head_dim);
            let mut data_ref = data.clone();
            unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
            scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
            assert!(vecs_approx_eq(&data, &data_ref, EPS), "mismatch at pos {pos}");
        }
    }

    #[test]
    fn test_sweep_positions_dim256() {
        let head_dim = 256;
        let max_pos = 8;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, max_pos, BASE);
        for pos in 0..max_pos {
            let mut data = make_data(head_dim);
            let mut data_ref = data.clone();
            unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
            scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
            assert!(vecs_approx_eq(&data, &data_ref, EPS), "mismatch at pos {pos}");
        }
    }

    // ── Sweep NeoX positions ────────────────────────────────────────

    #[test]
    fn test_sweep_neox_positions_dim64() {
        let head_dim = 64;
        let max_pos = 20;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, max_pos, BASE);
        for pos in 0..max_pos {
            let mut data = make_data(head_dim);
            let mut data_ref = data.clone();
            unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
            scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
            assert!(vecs_approx_eq(&data, &data_ref, EPS), "neox mismatch at pos {pos}");
        }
    }

    // ── Various base values ─────────────────────────────────────────

    #[test]
    fn test_base_100() {
        let base = 100.0;
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, base);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, 3, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, 3, base);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_base_1000000() {
        let base = 1_000_000.0;
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, base);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, 5, head_dim) };
        scalar_rope_apply(&mut data_ref, head_dim, 5, base);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    // ── Cache reuse across calls ────────────────────────────────────

    #[test]
    fn test_cache_reuse() {
        let head_dim = 64;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 32, BASE);

        // Use same cache for multiple independent calls
        for pos in [0, 5, 15, 31] {
            let mut data = make_data(head_dim);
            let mut data_ref = data.clone();
            unsafe { neon_rope_apply_cached_f32(&mut data, &cos, &sin, pos, head_dim) };
            scalar_rope_apply(&mut data_ref, head_dim, pos, BASE);
            assert!(vecs_approx_eq(&data, &data_ref, EPS), "cache reuse fail at pos {pos}");
        }
    }

    // ── Pair-level rotation verification ────────────────────────────

    #[test]
    fn test_single_pair_rotation() {
        // Manually verify the rotation formula on a single 2-element head
        let head_dim = 2;
        let pos = 3;
        let x0 = 1.0f32;
        let x1 = 2.0f32;
        let angle = theta(pos, 0, head_dim, BASE);
        let expected_0 = x0 * angle.cos() - x1 * angle.sin();
        let expected_1 = x0 * angle.sin() + x1 * angle.cos();

        let mut q = vec![x0, x1];
        let mut k = vec![0.0, 0.0];
        unsafe { neon_rope_apply_f32(&mut q, &mut k, head_dim, pos, BASE) };
        assert!(approx_eq(q[0], expected_0, EPS));
        assert!(approx_eq(q[1], expected_1, EPS));
    }

    // ── Batch with odd head counts / seq lens ───────────────────────

    #[test]
    fn test_batch_1head_1seq() {
        let head_dim = 32;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 1, BASE);
        let mut data = make_data(head_dim);
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_batch_f32(&mut data, &cos, &sin, 1, 1, 1, head_dim) };
        unsafe { neon_rope_apply_cached_f32(&mut data_ref, &cos, &sin, 0, head_dim) };
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    #[test]
    fn test_batch_3heads() {
        let head_dim = 32;
        let num_heads = 3;
        let seq_len = 2;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, seq_len, BASE);
        let total = num_heads * seq_len * head_dim;
        let mut batch_data = make_data(total);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(&mut batch_data, &cos, &sin, 1, num_heads, seq_len, head_dim);
        }
        for h in 0..num_heads {
            for s in 0..seq_len {
                let off = (h * seq_len + s) * head_dim;
                unsafe {
                    neon_rope_apply_cached_f32(
                        &mut seq_data[off..off + head_dim],
                        &cos,
                        &sin,
                        s,
                        head_dim,
                    );
                }
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    #[test]
    fn test_batch_5seq() {
        let head_dim = 32;
        let num_heads = 2;
        let seq_len = 5;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, seq_len, BASE);
        let total = num_heads * seq_len * head_dim;
        let mut batch_data = make_data(total);
        let mut seq_data = batch_data.clone();

        unsafe {
            neon_rope_apply_batch_f32(&mut batch_data, &cos, &sin, 1, num_heads, seq_len, head_dim);
        }
        for h in 0..num_heads {
            for s in 0..seq_len {
                let off = (h * seq_len + s) * head_dim;
                unsafe {
                    neon_rope_apply_cached_f32(
                        &mut seq_data[off..off + head_dim],
                        &cos,
                        &sin,
                        s,
                        head_dim,
                    );
                }
            }
        }
        assert!(vecs_approx_eq(&batch_data, &seq_data, EPS));
    }

    // ── Apply twice vs single application at summed position ────────

    #[test]
    fn test_double_apply_pair_equivalence() {
        // For a single pair, applying at pos=a then pos=b should equal
        // applying at pos=a+b because rotation angles add.
        let head_dim = 2;
        let a = 3usize;
        let b = 5usize;

        let mut twice = vec![1.0f32, 2.0];
        let mut k_dummy = [0.0; 2];
        unsafe { neon_rope_apply_f32(&mut twice, &mut k_dummy, head_dim, a, BASE) };
        unsafe { neon_rope_apply_f32(&mut twice, &mut k_dummy, head_dim, b, BASE) };

        let mut once = vec![1.0f32, 2.0];
        unsafe { neon_rope_apply_f32(&mut once, &mut k_dummy, head_dim, a + b, BASE) };

        assert!(vecs_approx_eq(&twice, &once, EPS));
    }

    #[test]
    fn test_double_apply_pair_equivalence_cached() {
        let head_dim = 2;
        let a = 2usize;
        let b = 4usize;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, a + b + 1, BASE);

        let mut twice = vec![3.0f32, 7.0];
        unsafe { neon_rope_apply_cached_f32(&mut twice, &cos, &sin, a, head_dim) };
        unsafe { neon_rope_apply_cached_f32(&mut twice, &cos, &sin, b, head_dim) };

        let mut once = vec![3.0f32, 7.0];
        unsafe { neon_rope_apply_cached_f32(&mut once, &cos, &sin, a + b, head_dim) };

        assert!(vecs_approx_eq(&twice, &once, EPS));
    }

    // ── Negative values ─────────────────────────────────────────────

    #[test]
    fn test_negative_values() {
        let head_dim = 32;
        let pos = 4;
        let mut q: Vec<f32> = (0..head_dim as i32).map(|i| -(i as f32) * 0.3).collect();
        let mut k = q.clone();
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, head_dim, pos, BASE) };
        scalar_rope_apply(&mut q_ref, head_dim, pos, BASE);
        scalar_rope_apply(&mut k_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    #[test]
    fn test_negative_values_neox() {
        let head_dim = 32;
        let pos = 4;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 8, BASE);
        let mut data: Vec<f32> = (0..head_dim as i32).map(|i| -(i as f32) * 0.3).collect();
        let mut data_ref = data.clone();
        unsafe { neon_rope_apply_neox_f32(&mut data, &cos, &sin, pos, head_dim) };
        scalar_rope_apply_neox(&mut data_ref, &cos, &sin, pos, head_dim);
        assert!(vecs_approx_eq(&data, &data_ref, EPS));
    }

    // ── Large values ────────────────────────────────────────────────

    #[test]
    fn test_large_values() {
        let head_dim = 32;
        let pos = 2;
        let mut q = vec![1e6f32; head_dim];
        let mut k = vec![1e6f32; head_dim];
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, head_dim, pos, BASE) };
        scalar_rope_apply(&mut q_ref, head_dim, pos, BASE);
        scalar_rope_apply(&mut k_ref, head_dim, pos, BASE);
        // Relax tolerance for large values
        assert!(vecs_approx_eq(&q, &q_ref, 1.0));
        assert!(vecs_approx_eq(&k, &k_ref, 1.0));
    }

    // ── Alternating pattern ─────────────────────────────────────────

    #[test]
    fn test_alternating_pattern() {
        let head_dim = 64;
        let pos = 6;
        let mut q: Vec<f32> = (0..head_dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut k = q.clone();
        let mut q_ref = q.clone();
        let mut k_ref = k.clone();
        unsafe { neon_rope_apply_f32(&mut q, &mut k, head_dim, pos, BASE) };
        scalar_rope_apply(&mut q_ref, head_dim, pos, BASE);
        scalar_rope_apply(&mut k_ref, head_dim, pos, BASE);
        assert!(vecs_approx_eq(&q, &q_ref, EPS));
        assert!(vecs_approx_eq(&k, &k_ref, EPS));
    }

    // ── Direct vs cached agreement for Q/K ──────────────────────────

    #[test]
    fn test_apply_f32_agrees_with_cached_q() {
        let head_dim = 64;
        let pos = 9;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);

        let orig = make_data(head_dim);
        let mut direct_q = orig.clone();
        let mut direct_k = make_data(head_dim);
        let mut cached = orig.clone();

        unsafe { neon_rope_apply_f32(&mut direct_q, &mut direct_k, head_dim, pos, BASE) };
        unsafe { neon_rope_apply_cached_f32(&mut cached, &cos, &sin, pos, head_dim) };
        assert!(vecs_approx_eq(&direct_q, &cached, EPS));
    }

    #[test]
    fn test_apply_f32_agrees_with_cached_k() {
        let head_dim = 64;
        let pos = 9;
        let (cos, sin) = neon_rope_build_cos_sin_cache(head_dim, 16, BASE);

        let orig = make_data(head_dim);
        let mut direct_q = make_data(head_dim);
        let mut direct_k = orig.clone();
        let mut cached = orig.clone();

        unsafe { neon_rope_apply_f32(&mut direct_q, &mut direct_k, head_dim, pos, BASE) };
        unsafe { neon_rope_apply_cached_f32(&mut cached, &cos, &sin, pos, head_dim) };
        assert!(vecs_approx_eq(&direct_k, &cached, EPS));
    }
}
