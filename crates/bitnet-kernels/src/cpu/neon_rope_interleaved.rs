//! ARM NEON interleaved RoPE (Rotary Position Embedding) for Apple Silicon.
//!
//! Two rotation layouts:
//! - **Interleaved**: pairs `(x0,x1),(x2,x3),...` sit adjacent in memory.
//! - **Half-rotary**: first half and second half of the vector form pairs.
//!
//! Uses `vld2q_f32` for structure-of-arrays loads and `vfmaq_f32` for fused
//! multiply-add where beneficial.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Cache Construction ──────────────────────────────────────────────

/// Build cos/sin frequency caches for RoPE.
///
/// Returns `(cos_cache, sin_cache)` each of length `max_seq_len * half_dim`
/// where `half_dim = head_dim / 2`. Layout: `cache[pos * half_dim + i]`.
///
/// Uses the standard RoPE formula: `theta_i = base^(-2i / head_dim)`,
/// `angle = pos * theta_i`.
pub fn neon_build_rope_cache(
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
            let exponent = -(2.0 * i as f32) / head_dim as f32;
            let theta = base.powf(exponent);
            let angle = pos as f32 * theta;
            cos_cache.push(angle.cos());
            sin_cache.push(angle.sin());
        }
    }

    (cos_cache, sin_cache)
}

// ── Interleaved RoPE ────────────────────────────────────────────────

/// Apply RoPE with interleaved pair layout **in-place** using NEON.
///
/// Interleaved layout: `[x0, x1, x2, x3, ...]` where `(x0, x1)` and
/// `(x2, x3)` are successive rotation pairs.
///
/// `cos_cache` / `sin_cache` must cover `seq_pos` (i.e. at least
/// `(seq_pos + 1) * head_dim / 2` entries).
///
/// # Panics
///
/// Panics if `x` is shorter than `head_dim` or caches are too small.
#[cfg(target_arch = "aarch64")]
pub fn neon_rope_interleaved(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    assert!(x.len() >= head_dim, "x too short for head_dim");
    let half_dim = head_dim / 2;
    let table_offset = seq_pos * half_dim;
    assert!(cos_cache.len() >= table_offset + half_dim, "cos_cache too small for seq_pos");
    assert!(sin_cache.len() >= table_offset + half_dim, "sin_cache too small for seq_pos");

    // NEON path: vld2q_f32 deinterleaves 8 floats into two 4-wide registers,
    // giving us evens (x0,x2,x4,x6) and odds (x1,x3,x5,x7) — exactly the
    // real and imaginary parts of 4 rotation pairs at once.
    let neon_pairs = half_dim / 4;
    unsafe {
        for c in 0..neon_pairs {
            let data_idx = c * 8;
            let tbl_idx = table_offset + c * 4;

            // Deinterleave load: evens → val.0, odds → val.1
            let val = vld2q_f32(x.as_ptr().add(data_idx));
            let re = val.0; // x0, x2, x4, x6
            let im = val.1; // x1, x3, x5, x7

            let cos_v = vld1q_f32(cos_cache.as_ptr().add(tbl_idx));
            let sin_v = vld1q_f32(sin_cache.as_ptr().add(tbl_idx));

            // Rotation: re' = re * cos - im * sin
            //           im' = re * sin + im * cos
            let neg_sin = vnegq_f32(sin_v);
            let new_re = vfmaq_f32(vmulq_f32(re, cos_v), im, neg_sin);
            let new_im = vfmaq_f32(vmulq_f32(im, cos_v), re, sin_v);

            let out = float32x4x2_t(new_re, new_im);
            vst2q_f32(x.as_mut_ptr().add(data_idx), out);
        }
    }

    // Scalar tail for remaining pairs.
    let processed = neon_pairs * 4;
    for i in processed..half_dim {
        let idx = i * 2;
        let cos_val = cos_cache[table_offset + i];
        let sin_val = sin_cache[table_offset + i];
        let x0 = x[idx];
        let x1 = x[idx + 1];
        x[idx] = x0 * cos_val - x1 * sin_val;
        x[idx + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// ── Half-Rotary RoPE ────────────────────────────────────────────────

/// Apply RoPE with half-rotary layout **in-place** using NEON.
///
/// Half-rotary layout: the first `head_dim/2` elements are the "real" parts
/// and the second `head_dim/2` elements are the "imaginary" parts. Pair `i`
/// is `(x[i], x[i + half_dim])`.
///
/// # Panics
///
/// Panics if `x` is shorter than `head_dim` or caches are too small.
#[cfg(target_arch = "aarch64")]
pub fn neon_rope_half_rotary(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    assert!(x.len() >= head_dim, "x too short for head_dim");
    let half_dim = head_dim / 2;
    let table_offset = seq_pos * half_dim;
    assert!(cos_cache.len() >= table_offset + half_dim, "cos_cache too small for seq_pos");
    assert!(sin_cache.len() >= table_offset + half_dim, "sin_cache too small for seq_pos");

    // NEON: process 4 pairs per iteration.
    let neon_iters = half_dim / 4;
    unsafe {
        for c in 0..neon_iters {
            let offset = c * 4;
            let tbl_idx = table_offset + offset;

            let re = vld1q_f32(x.as_ptr().add(offset));
            let im = vld1q_f32(x.as_ptr().add(half_dim + offset));

            let cos_v = vld1q_f32(cos_cache.as_ptr().add(tbl_idx));
            let sin_v = vld1q_f32(sin_cache.as_ptr().add(tbl_idx));

            let neg_sin = vnegq_f32(sin_v);
            let new_re = vfmaq_f32(vmulq_f32(re, cos_v), im, neg_sin);
            let new_im = vfmaq_f32(vmulq_f32(im, cos_v), re, sin_v);

            vst1q_f32(x.as_mut_ptr().add(offset), new_re);
            vst1q_f32(x.as_mut_ptr().add(half_dim + offset), new_im);
        }
    }

    // Scalar tail.
    let processed = neon_iters * 4;
    for i in processed..half_dim {
        let cos_val = cos_cache[table_offset + i];
        let sin_val = sin_cache[table_offset + i];
        let re = x[i];
        let im = x[half_dim + i];
        x[i] = re * cos_val - im * sin_val;
        x[half_dim + i] = re * sin_val + im * cos_val;
    }
}

// ── Batch Application ───────────────────────────────────────────────

/// Apply interleaved RoPE to a batch of sequences.
///
/// `x` layout: `[batch_size × head_dim]` — each contiguous `head_dim` block
/// is one sequence element. Positions are `start_pos, start_pos+1, ...`.
///
/// # Panics
///
/// Panics if `x.len() < batch_size * head_dim` or caches are too small.
#[cfg(target_arch = "aarch64")]
pub fn neon_rope_apply_batch(
    x: &mut [f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    batch_size: usize,
    head_dim: usize,
    start_pos: usize,
) {
    assert!(x.len() >= batch_size * head_dim, "x too short for batch_size * head_dim");
    for b in 0..batch_size {
        let offset = b * head_dim;
        let pos = start_pos + b;
        neon_rope_interleaved(
            &mut x[offset..offset + head_dim],
            cos_cache,
            sin_cache,
            head_dim,
            pos,
        );
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    const BASE: f32 = 10_000.0;

    #[test]
    fn test_rope_cache_basic() {
        let head_dim = 8;
        let max_seq = 4;
        let (cos_c, sin_c) = neon_build_rope_cache(head_dim, max_seq, BASE);

        let half_dim = head_dim / 2;
        assert_eq!(cos_c.len(), max_seq * half_dim);
        assert_eq!(sin_c.len(), max_seq * half_dim);

        // At position 0 every angle is 0 → cos=1, sin=0.
        for i in 0..half_dim {
            assert!((cos_c[i] - 1.0).abs() < 1e-6, "cos[{i}] = {}", cos_c[i]);
            assert!(sin_c[i].abs() < 1e-6, "sin[{i}] = {}", sin_c[i]);
        }

        // At position 1, dimension-pair 0: theta = base^0 = 1, angle = 1.
        let idx = 1 * half_dim; // pos=1, i=0
        let expected_cos = 1.0_f32.cos();
        let expected_sin = 1.0_f32.sin();
        assert!((cos_c[idx] - expected_cos).abs() < 1e-6);
        assert!((sin_c[idx] - expected_sin).abs() < 1e-6);
    }

    #[test]
    fn test_interleaved_rope_identity_at_pos0() {
        let head_dim = 16;
        let (cos_c, sin_c) = neon_build_rope_cache(head_dim, 1, BASE);

        let original: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
        let mut x = original.clone();

        neon_rope_interleaved(&mut x, &cos_c, &sin_c, head_dim, 0);

        // cos=1, sin=0 at pos 0 ⇒ output should equal input.
        for i in 0..head_dim {
            assert!(
                (x[i] - original[i]).abs() < 1e-5,
                "mismatch at {i}: got {} expected {}",
                x[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_half_rotary_vs_interleaved() {
        let head_dim = 8;
        let max_seq = 4;
        let (cos_c, sin_c) = neon_build_rope_cache(head_dim, max_seq, BASE);
        let half_dim = head_dim / 2;

        for pos in 0..max_seq {
            // Build identical logical input in both layouts.
            let re: Vec<f32> = (0..half_dim).map(|i| (i + 1) as f32).collect();
            let im: Vec<f32> = (0..half_dim).map(|i| (i + half_dim + 1) as f32).collect();

            // Interleaved: [re0, im0, re1, im1, ...]
            let mut interleaved = vec![0.0f32; head_dim];
            for i in 0..half_dim {
                interleaved[2 * i] = re[i];
                interleaved[2 * i + 1] = im[i];
            }

            // Half-rotary: [re0, re1, ..., im0, im1, ...]
            let mut half_rot = vec![0.0f32; head_dim];
            half_rot[..half_dim].copy_from_slice(&re);
            half_rot[half_dim..].copy_from_slice(&im);

            neon_rope_interleaved(&mut interleaved, &cos_c, &sin_c, head_dim, pos);
            neon_rope_half_rotary(&mut half_rot, &cos_c, &sin_c, head_dim, pos);

            // Compare: both should produce the same rotation per pair.
            for i in 0..half_dim {
                let int_re = interleaved[2 * i];
                let int_im = interleaved[2 * i + 1];
                let hr_re = half_rot[i];
                let hr_im = half_rot[half_dim + i];
                assert!(
                    (int_re - hr_re).abs() < 1e-5,
                    "pos={pos} pair={i} re: interleaved={int_re} half_rot={hr_re}"
                );
                assert!(
                    (int_im - hr_im).abs() < 1e-5,
                    "pos={pos} pair={i} im: interleaved={int_im} half_rot={hr_im}"
                );
            }
        }
    }

    #[test]
    fn test_rope_batch_consistency() {
        let head_dim = 8;
        let batch_size = 3;
        let start_pos = 2;
        let max_seq = start_pos + batch_size;
        let (cos_c, sin_c) = neon_build_rope_cache(head_dim, max_seq, BASE);

        // Prepare batch input and a clone for individual applies.
        let input: Vec<f32> = (0..batch_size * head_dim).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let mut batched = input.clone();
        let mut individual = input.clone();

        neon_rope_apply_batch(&mut batched, &cos_c, &sin_c, batch_size, head_dim, start_pos);

        for b in 0..batch_size {
            let off = b * head_dim;
            neon_rope_interleaved(
                &mut individual[off..off + head_dim],
                &cos_c,
                &sin_c,
                head_dim,
                start_pos + b,
            );
        }

        for i in 0..batched.len() {
            assert!(
                (batched[i] - individual[i]).abs() < 1e-5,
                "mismatch at {i}: batch={} individual={}",
                batched[i],
                individual[i]
            );
        }
    }
}
