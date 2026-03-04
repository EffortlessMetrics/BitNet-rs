#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]
//! NEON-optimized RoPE (Rotary Position Embedding) v3 kernel for Apple Silicon.
//!
//! Provides six RoPE operations with NEON SIMD acceleration on AArch64:
//! - Forward rotation (standard paired layout)
//! - Cos/sin table precomputation
//! - Batch rotation across heads and positions
//! - NeoX-style rotation (split-half layout)
//! - Scaled rotation for extended context (YaRN/NTK-aware)
//! - Inverse rotation for KV cache reuse
//!
//! Each operation has an `unsafe fn neon_*` variant, a safe `scalar_*` fallback,
//! and a public dispatcher that selects NEON at runtime via
//! `is_aarch64_feature_detected!("neon")`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Scalar fallbacks ────────────────────────────────────────────────────

/// Scalar RoPE forward rotation in-place.
pub fn scalar_rope_forward_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    for i in 0..half {
        let x0 = input[2 * i];
        let x1 = input[2 * i + 1];
        let c = cos_table[offset + i];
        let s = sin_table[offset + i];
        input[2 * i] = x0 * c - x1 * s;
        input[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// Scalar cos/sin table builder.
pub fn scalar_build_cos_sin_tables(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) {
    let half = head_dim / 2;
    for pos in 0..max_seq_len {
        for i in 0..half {
            let exponent = -(2.0 * i as f32) / head_dim as f32;
            let theta = base.powf(exponent);
            let angle = pos as f32 * theta;
            cos_out[pos * half + i] = angle.cos();
            sin_out[pos * half + i] = angle.sin();
        }
    }
}

/// Scalar batch RoPE across heads and positions.
pub fn scalar_rope_batch_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_start: usize,
    seq_len: usize,
) {
    for s in 0..seq_len {
        let pos = seq_start + s;
        for h in 0..num_heads {
            let base = (s * num_heads + h) * head_dim;
            scalar_rope_forward_f32(
                &mut input[base..base + head_dim],
                cos_table,
                sin_table,
                head_dim,
                pos,
            );
        }
    }
}

/// Scalar NeoX-style RoPE (split-half layout).
///
/// NeoX splits the vector into first-half and second-half instead of
/// interleaved pairs: pairs are (x[i], x[i + half]).
pub fn scalar_rope_neox_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    for i in 0..half {
        let x0 = input[i];
        let x1 = input[i + half];
        let c = cos_table[offset + i];
        let s = sin_table[offset + i];
        input[i] = x0 * c - x1 * s;
        input[i + half] = x0 * s + x1 * c;
    }
}

/// Scalar scaled RoPE for extended context windows.
pub fn scalar_rope_with_scaling_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
    scale: f32,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    for i in 0..half {
        let x0 = input[2 * i];
        let x1 = input[2 * i + 1];
        let c = cos_table[offset + i] * scale;
        let s = sin_table[offset + i] * scale;
        input[2 * i] = x0 * c - x1 * s;
        input[2 * i + 1] = x0 * s + x1 * c;
    }
}

/// Scalar inverse RoPE rotation (negate sin to reverse).
pub fn scalar_inverse_rope_f32(
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    for i in 0..half {
        let x0 = output[2 * i];
        let x1 = output[2 * i + 1];
        let c = cos_table[offset + i];
        let s = sin_table[offset + i];
        // Inverse rotation: negate sin component
        output[2 * i] = x0 * c + x1 * s;
        output[2 * i + 1] = -x0 * s + x1 * c;
    }
}

// ── NEON implementations ────────────────────────────────────────────────

/// NEON-accelerated RoPE forward rotation in-place.
///
/// # Safety
/// Requires AArch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_forward_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    // Process 2 pairs (4 floats) at a time
    let chunks = half / 2;
    let sign = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());

    for c in 0..chunks {
        let di = c * 4;
        let ti = offset + c * 2;

        let vals = vld1q_f32(input.as_ptr().add(di));
        let swapped = vrev64q_f32(vals);

        let c0 = *cos_table.get_unchecked(ti);
        let c1 = *cos_table.get_unchecked(ti + 1);
        let s0 = *sin_table.get_unchecked(ti);
        let s1 = *sin_table.get_unchecked(ti + 1);

        let cos_v = vld1q_f32([c0, c0, c1, c1].as_ptr());
        let sin_v = vld1q_f32([s0, s0, s1, s1].as_ptr());

        // result = vals * cos + swapped * sign * sin  (FMA)
        let prod = vmulq_f32(vmulq_f32(swapped, sign), sin_v);
        let rotated = vfmaq_f32(prod, vals, cos_v);

        vst1q_f32(input.as_mut_ptr().add(di), rotated);
    }

    // Scalar tail
    let done = chunks * 2;
    for i in done..half {
        let idx = i * 2;
        let c = *cos_table.get_unchecked(offset + i);
        let s = *sin_table.get_unchecked(offset + i);
        let x0 = input[idx];
        let x1 = input[idx + 1];
        input[idx] = x0 * c - x1 * s;
        input[idx + 1] = x0 * s + x1 * c;
    }
}

/// NEON-accelerated cos/sin table builder.
///
/// # Safety
/// Requires AArch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_build_cos_sin_tables(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) {
    let half = head_dim / 2;
    // Precompute theta values
    let mut thetas = vec![0.0f32; half];
    for i in 0..half {
        let exponent = -(2.0 * i as f32) / head_dim as f32;
        thetas[i] = base.powf(exponent);
    }

    for pos in 0..max_seq_len {
        let row = pos * half;
        let pos_f = pos as f32;
        let chunks = half / 4;
        for c in 0..chunks {
            let ti = c * 4;
            let a0 = pos_f * thetas[ti];
            let a1 = pos_f * thetas[ti + 1];
            let a2 = pos_f * thetas[ti + 2];
            let a3 = pos_f * thetas[ti + 3];
            // Store cos
            let cos_v = vld1q_f32([a0.cos(), a1.cos(), a2.cos(), a3.cos()].as_ptr());
            vst1q_f32(cos_out.as_mut_ptr().add(row + ti), cos_v);
            // Store sin
            let sin_v = vld1q_f32([a0.sin(), a1.sin(), a2.sin(), a3.sin()].as_ptr());
            vst1q_f32(sin_out.as_mut_ptr().add(row + ti), sin_v);
        }
        // Scalar tail
        for i in (chunks * 4)..half {
            let angle = pos_f * thetas[i];
            cos_out[row + i] = angle.cos();
            sin_out[row + i] = angle.sin();
        }
    }
}

/// NEON-accelerated batch RoPE across heads and positions.
///
/// # Safety
/// Requires AArch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_batch_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_start: usize,
    seq_len: usize,
) {
    for s in 0..seq_len {
        let pos = seq_start + s;
        for h in 0..num_heads {
            let base = (s * num_heads + h) * head_dim;
            neon_rope_forward_f32(
                &mut input[base..base + head_dim],
                cos_table,
                sin_table,
                head_dim,
                pos,
            );
        }
    }
}

/// NEON-accelerated NeoX-style RoPE (split-half layout).
///
/// # Safety
/// Requires AArch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_neox_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    let chunks = half / 4;

    for c in 0..chunks {
        let i = c * 4;
        let ti = offset + i;

        let first = vld1q_f32(input.as_ptr().add(i));
        let second = vld1q_f32(input.as_ptr().add(i + half));

        let cos_v = vld1q_f32(cos_table.as_ptr().add(ti));
        let sin_v = vld1q_f32(sin_table.as_ptr().add(ti));

        // first_out  = first * cos - second * sin
        let neg_sin = vnegq_f32(sin_v);
        let first_out = vfmaq_f32(vmulq_f32(second, neg_sin), first, cos_v);
        // second_out = first * sin + second * cos
        let second_out = vfmaq_f32(vmulq_f32(first, sin_v), second, cos_v);

        vst1q_f32(input.as_mut_ptr().add(i), first_out);
        vst1q_f32(input.as_mut_ptr().add(i + half), second_out);
    }

    // Scalar tail
    for i in (chunks * 4)..half {
        let ti = offset + i;
        let x0 = input[i];
        let x1 = input[i + half];
        let c = cos_table[ti];
        let s = sin_table[ti];
        input[i] = x0 * c - x1 * s;
        input[i + half] = x0 * s + x1 * c;
    }
}

/// NEON-accelerated scaled RoPE for extended context.
///
/// # Safety
/// Requires AArch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_rope_with_scaling_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
    scale: f32,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    let chunks = half / 2;
    let sign = vld1q_f32([-1.0f32, 1.0, -1.0, 1.0].as_ptr());
    let scale_v = vdupq_n_f32(scale);

    for c in 0..chunks {
        let di = c * 4;
        let ti = offset + c * 2;

        let vals = vld1q_f32(input.as_ptr().add(di));
        let swapped = vrev64q_f32(vals);

        let c0 = *cos_table.get_unchecked(ti);
        let c1 = *cos_table.get_unchecked(ti + 1);
        let s0 = *sin_table.get_unchecked(ti);
        let s1 = *sin_table.get_unchecked(ti + 1);

        let cos_v = vmulq_f32(vld1q_f32([c0, c0, c1, c1].as_ptr()), scale_v);
        let sin_v = vmulq_f32(vld1q_f32([s0, s0, s1, s1].as_ptr()), scale_v);

        let prod = vmulq_f32(vmulq_f32(swapped, sign), sin_v);
        let rotated = vfmaq_f32(prod, vals, cos_v);

        vst1q_f32(input.as_mut_ptr().add(di), rotated);
    }

    // Scalar tail
    let done = chunks * 2;
    for i in done..half {
        let idx = i * 2;
        let c = *cos_table.get_unchecked(offset + i) * scale;
        let s = *sin_table.get_unchecked(offset + i) * scale;
        let x0 = input[idx];
        let x1 = input[idx + 1];
        input[idx] = x0 * c - x1 * s;
        input[idx + 1] = x0 * s + x1 * c;
    }
}

/// NEON-accelerated inverse RoPE rotation.
///
/// # Safety
/// Requires AArch64 NEON support.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_inverse_rope_f32(
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    let half = head_dim / 2;
    let offset = seq_pos * half;
    let chunks = half / 2;
    // Inverse uses [+1, -1, +1, -1] (negated sin component)
    let sign = vld1q_f32([1.0f32, -1.0, 1.0, -1.0].as_ptr());

    for c in 0..chunks {
        let di = c * 4;
        let ti = offset + c * 2;

        let vals = vld1q_f32(output.as_ptr().add(di));
        let swapped = vrev64q_f32(vals);

        let c0 = *cos_table.get_unchecked(ti);
        let c1 = *cos_table.get_unchecked(ti + 1);
        let s0 = *sin_table.get_unchecked(ti);
        let s1 = *sin_table.get_unchecked(ti + 1);

        let cos_v = vld1q_f32([c0, c0, c1, c1].as_ptr());
        let sin_v = vld1q_f32([s0, s0, s1, s1].as_ptr());

        // inverse: vals * cos + swapped * [+1,-1,+1,-1] * sin
        let prod = vmulq_f32(vmulq_f32(swapped, sign), sin_v);
        let rotated = vfmaq_f32(prod, vals, cos_v);

        vst1q_f32(output.as_mut_ptr().add(di), rotated);
    }

    // Scalar tail
    let done = chunks * 2;
    for i in done..half {
        let idx = i * 2;
        let c = *cos_table.get_unchecked(offset + i);
        let s = *sin_table.get_unchecked(offset + i);
        let x0 = output[idx];
        let x1 = output[idx + 1];
        output[idx] = x0 * c + x1 * s;
        output[idx + 1] = -x0 * s + x1 * c;
    }
}

// ── Public dispatchers ──────────────────────────────────────────────────

/// Apply RoPE forward rotation in-place, dispatching to NEON when available.
pub fn rope_forward_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_rope_forward_f32(input, cos_table, sin_table, head_dim, seq_pos);
            }
            return;
        }
    }
    scalar_rope_forward_f32(input, cos_table, sin_table, head_dim, seq_pos);
}

/// Precompute cos/sin frequency tables, dispatching to NEON when available.
pub fn build_cos_sin_tables(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_build_cos_sin_tables(head_dim, max_seq_len, base, cos_out, sin_out);
            }
            return;
        }
    }
    scalar_build_cos_sin_tables(head_dim, max_seq_len, base, cos_out, sin_out);
}

/// Batch RoPE for multiple heads/positions, dispatching to NEON when available.
pub fn rope_batch_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_start: usize,
    seq_len: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_rope_batch_f32(
                    input, cos_table, sin_table, num_heads, head_dim, seq_start, seq_len,
                );
            }
            return;
        }
    }
    scalar_rope_batch_f32(input, cos_table, sin_table, num_heads, head_dim, seq_start, seq_len);
}

/// NeoX-style RoPE (split-half layout), dispatching to NEON when available.
pub fn rope_neox_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_rope_neox_f32(input, cos_table, sin_table, head_dim, seq_pos);
            }
            return;
        }
    }
    scalar_rope_neox_f32(input, cos_table, sin_table, head_dim, seq_pos);
}

/// Scaled RoPE for extended context, dispatching to NEON when available.
pub fn rope_with_scaling_f32(
    input: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_rope_with_scaling_f32(input, cos_table, sin_table, head_dim, seq_pos, scale);
            }
            return;
        }
    }
    scalar_rope_with_scaling_f32(input, cos_table, sin_table, head_dim, seq_pos, scale);
}

/// Inverse RoPE rotation for KV cache reuse, dispatching to NEON when available.
pub fn inverse_rope_f32(
    output: &mut [f32],
    cos_table: &[f32],
    sin_table: &[f32],
    head_dim: usize,
    seq_pos: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_inverse_rope_f32(output, cos_table, sin_table, head_dim, seq_pos);
            }
            return;
        }
    }
    scalar_inverse_rope_f32(output, cos_table, sin_table, head_dim, seq_pos);
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn make_tables(head_dim: usize, max_seq: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
        let half = head_dim / 2;
        let len = max_seq * half;
        let mut cos_t = vec![0.0f32; len];
        let mut sin_t = vec![0.0f32; len];
        scalar_build_cos_sin_tables(head_dim, max_seq, base, &mut cos_t, &mut sin_t);
        (cos_t, sin_t)
    }

    // ── Table correctness ───────────────────────────────────────────

    #[test]
    fn test_table_pos0_is_cos0_sin0() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        // pos=0 → angle=0 for all dims → cos=1, sin=0
        for i in 0..4 {
            assert!(approx_eq(cos_t[i], 1.0), "cos[{i}]={}", cos_t[i]);
            assert!(approx_eq(sin_t[i], 0.0), "sin[{i}]={}", sin_t[i]);
        }
    }

    #[test]
    fn test_table_known_values_dim0() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let half = 4;
        // pos=1, dim_pair=0: theta = 10000^0 = 1.0, angle = 1.0
        let idx = 1 * half + 0;
        assert!(approx_eq(cos_t[idx], 1.0f32.cos()));
        assert!(approx_eq(sin_t[idx], 1.0f32.sin()));
    }

    #[test]
    fn test_table_known_values_dim1() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let half = 4;
        // pos=1, dim_pair=1: theta = 10000^(-2/8) = 10000^(-0.25)
        let theta = 10000.0f32.powf(-0.25);
        let idx = 1 * half + 1;
        assert!(approx_eq(cos_t[idx], theta.cos()));
        assert!(approx_eq(sin_t[idx], theta.sin()));
    }

    #[test]
    fn test_table_pos2_dim2() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let half = 4;
        let theta = 10000.0f32.powf(-0.5);
        let angle = 2.0 * theta;
        let idx = 2 * half + 2;
        assert!(approx_eq(cos_t[idx], angle.cos()));
        assert!(approx_eq(sin_t[idx], angle.sin()));
    }

    #[test]
    fn test_table_symmetry() {
        let (cos_t, sin_t) = make_tables(16, 8, 10000.0);
        // cos²+sin²=1 for all entries
        for i in 0..cos_t.len() {
            let sum = cos_t[i] * cos_t[i] + sin_t[i] * sin_t[i];
            assert!((sum - 1.0).abs() < 1e-4, "entry {i}: sum={sum}");
        }
    }

    #[test]
    fn test_build_tables_dispatcher_matches_scalar() {
        let head_dim = 16;
        let max_seq = 8;
        let base = 10000.0;
        let half = head_dim / 2;
        let len = max_seq * half;
        let mut cos_d = vec![0.0f32; len];
        let mut sin_d = vec![0.0f32; len];
        build_cos_sin_tables(head_dim, max_seq, base, &mut cos_d, &mut sin_d);
        let (cos_s, sin_s) = make_tables(head_dim, max_seq, base);
        for i in 0..len {
            assert!(approx_eq(cos_d[i], cos_s[i]), "cos mismatch at {i}");
            assert!(approx_eq(sin_d[i], sin_s[i]), "sin mismatch at {i}");
        }
    }

    #[test]
    fn test_table_different_base() {
        let (cos_a, _) = make_tables(8, 4, 10000.0);
        let (cos_b, _) = make_tables(8, 4, 500.0);
        // pos=1 dim=1 should differ
        assert!(!approx_eq(cos_a[4 + 1], cos_b[4 + 1]));
    }

    #[test]
    fn test_table_length() {
        let (cos_t, sin_t) = make_tables(16, 32, 10000.0);
        assert_eq!(cos_t.len(), 32 * 8);
        assert_eq!(sin_t.len(), 32 * 8);
    }

    // ── Forward rotation ────────────────────────────────────────────

    #[test]
    fn test_forward_pos0_identity() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 0);
        // pos=0 → cos=1, sin=0 → identity
        for i in 0..8 {
            assert!(approx_eq(data[i], orig[i]), "idx {i}");
        }
    }

    #[test]
    fn test_forward_scalar_basic() {
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let mut data = vec![1.0, 0.0, 0.0, 1.0];
        scalar_rope_forward_f32(&mut data, &cos_t, &sin_t, 4, 1);
        // Manually: pair0: (1*cos - 0*sin, 1*sin + 0*cos) = (cos, sin)
        let half = 2;
        let c0 = cos_t[half + 0];
        let s0 = sin_t[half + 0];
        assert!(approx_eq(data[0], c0));
        assert!(approx_eq(data[1], s0));
    }

    #[test]
    fn test_forward_dispatcher_matches_scalar() {
        let (cos_t, sin_t) = make_tables(16, 8, 10000.0);
        let orig: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_forward_f32(&mut disp, &cos_t, &sin_t, 16, 3);
        scalar_rope_forward_f32(&mut scal, &cos_t, &sin_t, 16, 3);
        for i in 0..16 {
            assert!(approx_eq(disp[i], scal[i]), "mismatch at {i}");
        }
    }

    #[test]
    fn test_forward_modifies_data() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 2);
        // At pos=2 values should change
        assert!(data != orig);
    }

    #[test]
    fn test_forward_preserves_norm_approx() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 1);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-3);
    }

    #[test]
    fn test_forward_different_positions_differ() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut d1 = orig.clone();
        let mut d2 = orig.clone();
        rope_forward_f32(&mut d1, &cos_t, &sin_t, 8, 1);
        rope_forward_f32(&mut d2, &cos_t, &sin_t, 8, 2);
        assert!(d1 != d2);
    }

    #[test]
    fn test_forward_head_dim_64() {
        let (cos_t, sin_t) = make_tables(64, 4, 10000.0);
        let orig: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_forward_f32(&mut disp, &cos_t, &sin_t, 64, 2);
        scalar_rope_forward_f32(&mut scal, &cos_t, &sin_t, 64, 2);
        for i in 0..64 {
            assert!(approx_eq(disp[i], scal[i]), "dim64 mismatch at {i}");
        }
    }

    #[test]
    fn test_forward_head_dim_128() {
        let (cos_t, sin_t) = make_tables(128, 4, 10000.0);
        let orig: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_forward_f32(&mut disp, &cos_t, &sin_t, 128, 1);
        scalar_rope_forward_f32(&mut scal, &cos_t, &sin_t, 128, 1);
        for i in 0..128 {
            assert!(approx_eq(disp[i], scal[i]), "dim128 mismatch at {i}");
        }
    }

    // ── Inverse(Forward(x)) ≈ x ────────────────────────────────────

    #[test]
    fn test_inverse_forward_roundtrip_dim8() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 2);
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 8, 2);
        for i in 0..8 {
            assert!(
                approx_eq(data[i], orig[i]),
                "roundtrip fail at {i}: {} vs {}",
                data[i],
                orig[i]
            );
        }
    }

    #[test]
    fn test_inverse_forward_roundtrip_dim16() {
        let (cos_t, sin_t) = make_tables(16, 8, 10000.0);
        let orig: Vec<f32> = (0..16).map(|i| i as f32 * 0.3 + 1.0).collect();
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 16, 5);
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 16, 5);
        for i in 0..16 {
            assert!(approx_eq(data[i], orig[i]), "roundtrip16 fail at {i}");
        }
    }

    #[test]
    fn test_inverse_forward_roundtrip_dim64() {
        let (cos_t, sin_t) = make_tables(64, 8, 10000.0);
        let orig: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 64, 7);
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 64, 7);
        for i in 0..64 {
            assert!(approx_eq(data[i], orig[i]), "roundtrip64 fail at {i}");
        }
    }

    #[test]
    fn test_inverse_forward_roundtrip_pos0() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![5.0, -3.0, 7.0, 1.0, -2.0, 4.0, 0.0, 9.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 0);
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 8, 0);
        for i in 0..8 {
            assert!(approx_eq(data[i], orig[i]));
        }
    }

    #[test]
    fn test_inverse_scalar_matches_dispatcher() {
        let (cos_t, sin_t) = make_tables(16, 8, 10000.0);
        let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect();
        let mut disp = input.clone();
        let mut scal = input.clone();
        inverse_rope_f32(&mut disp, &cos_t, &sin_t, 16, 4);
        scalar_inverse_rope_f32(&mut scal, &cos_t, &sin_t, 16, 4);
        for i in 0..16 {
            assert!(approx_eq(disp[i], scal[i]));
        }
    }

    // ── Batch tests ─────────────────────────────────────────────────

    #[test]
    fn test_batch_single_head_single_pos() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut batch = orig.clone();
        let mut single = orig.clone();
        rope_batch_f32(&mut batch, &cos_t, &sin_t, 1, 8, 1, 1);
        rope_forward_f32(&mut single, &cos_t, &sin_t, 8, 1);
        for i in 0..8 {
            assert!(approx_eq(batch[i], single[i]), "batch vs single at {i}");
        }
    }

    #[test]
    fn test_batch_multi_head() {
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let num_heads = 3;
        let head_dim = 4;
        let orig: Vec<f32> = (0..num_heads * head_dim).map(|i| i as f32).collect();
        let mut batch = orig.clone();
        rope_batch_f32(&mut batch, &cos_t, &sin_t, num_heads, head_dim, 2, 1);
        // Each head should match individual forward at pos=2
        for h in 0..num_heads {
            let start = h * head_dim;
            let mut single = orig[start..start + head_dim].to_vec();
            rope_forward_f32(&mut single, &cos_t, &sin_t, head_dim, 2);
            for d in 0..head_dim {
                assert!(approx_eq(batch[start + d], single[d]), "head {h} dim {d}");
            }
        }
    }

    #[test]
    fn test_batch_multi_position() {
        let (cos_t, sin_t) = make_tables(4, 8, 10000.0);
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let seq_start = 2;
        let total = seq_len * num_heads * head_dim;
        let orig: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let mut batch = orig.clone();
        rope_batch_f32(&mut batch, &cos_t, &sin_t, num_heads, head_dim, seq_start, seq_len);
        // Verify each (seq, head) independently
        for s in 0..seq_len {
            let pos = seq_start + s;
            for h in 0..num_heads {
                let base_idx = (s * num_heads + h) * head_dim;
                let mut single = orig[base_idx..base_idx + head_dim].to_vec();
                rope_forward_f32(&mut single, &cos_t, &sin_t, head_dim, pos);
                for d in 0..head_dim {
                    assert!(approx_eq(batch[base_idx + d], single[d]), "s={s} h={h} d={d}");
                }
            }
        }
    }

    #[test]
    fn test_batch_scalar_matches_dispatcher() {
        let (cos_t, sin_t) = make_tables(8, 8, 10000.0);
        let orig: Vec<f32> = (0..3 * 2 * 8).map(|i| i as f32 * 0.05).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_batch_f32(&mut disp, &cos_t, &sin_t, 2, 8, 1, 3);
        scalar_rope_batch_f32(&mut scal, &cos_t, &sin_t, 2, 8, 1, 3);
        for i in 0..disp.len() {
            assert!(approx_eq(disp[i], scal[i]), "batch match at {i}");
        }
    }

    #[test]
    fn test_batch_heads_independent() {
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let num_heads = 4;
        let head_dim = 4;
        // Set all heads to same values
        let head_vals = vec![1.0, 0.5, -1.0, 0.5];
        let mut data: Vec<f32> = (0..num_heads).flat_map(|_| head_vals.clone()).collect();
        rope_batch_f32(&mut data, &cos_t, &sin_t, num_heads, head_dim, 1, 1);
        // All heads should be identical since same input + same position
        for h in 1..num_heads {
            for d in 0..head_dim {
                assert!(approx_eq(data[d], data[h * head_dim + d]), "head {h} differs at dim {d}");
            }
        }
    }

    // ── NeoX vs standard ────────────────────────────────────────────

    #[test]
    fn test_neox_differs_from_standard() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut std_data = orig.clone();
        let mut neox_data = orig.clone();
        rope_forward_f32(&mut std_data, &cos_t, &sin_t, 8, 1);
        rope_neox_f32(&mut neox_data, &cos_t, &sin_t, 8, 1);
        // Should produce different results on non-trivial input
        assert!(std_data != neox_data, "NeoX should differ from standard");
    }

    #[test]
    fn test_neox_pos0_identity() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_neox_f32(&mut data, &cos_t, &sin_t, 8, 0);
        for i in 0..8 {
            assert!(approx_eq(data[i], orig[i]));
        }
    }

    #[test]
    fn test_neox_preserves_norm() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope_neox_f32(&mut data, &cos_t, &sin_t, 8, 2);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-3);
    }

    #[test]
    fn test_neox_scalar_matches_dispatcher() {
        let (cos_t, sin_t) = make_tables(16, 8, 10000.0);
        let orig: Vec<f32> = (0..16).map(|i| i as f32 * 0.2 - 1.0).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_neox_f32(&mut disp, &cos_t, &sin_t, 16, 3);
        scalar_rope_neox_f32(&mut scal, &cos_t, &sin_t, 16, 3);
        for i in 0..16 {
            assert!(approx_eq(disp[i], scal[i]), "neox mismatch at {i}");
        }
    }

    #[test]
    fn test_neox_split_half_layout() {
        // NeoX pairs (x[i], x[i+half]) instead of (x[2i], x[2i+1])
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let mut data = vec![1.0, 0.0, 0.0, 0.0]; // x[0]=1, x[1]=0, x[2]=0, x[3]=0
        scalar_rope_neox_f32(&mut data, &cos_t, &sin_t, 4, 1);
        // pair(0): x0=1, x1=0 (from input[0] and input[2])
        let half = 2;
        let c0 = cos_t[half + 0];
        let s0 = sin_t[half + 0];
        assert!(approx_eq(data[0], 1.0 * c0 - 0.0 * s0)); // first half
        assert!(approx_eq(data[2], 1.0 * s0 + 0.0 * c0)); // second half
    }

    #[test]
    fn test_neox_dim64() {
        let (cos_t, sin_t) = make_tables(64, 4, 10000.0);
        let orig: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_neox_f32(&mut disp, &cos_t, &sin_t, 64, 2);
        scalar_rope_neox_f32(&mut scal, &cos_t, &sin_t, 64, 2);
        for i in 0..64 {
            assert!(approx_eq(disp[i], scal[i]), "neox64 at {i}");
        }
    }

    // ── Scaling ─────────────────────────────────────────────────────

    #[test]
    fn test_scaling_factor_1_equals_forward() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut fwd = orig.clone();
        let mut scaled = orig.clone();
        rope_forward_f32(&mut fwd, &cos_t, &sin_t, 8, 1);
        rope_with_scaling_f32(&mut scaled, &cos_t, &sin_t, 8, 1, 1.0);
        for i in 0..8 {
            assert!(approx_eq(fwd[i], scaled[i]), "scale=1 at {i}");
        }
    }

    #[test]
    fn test_scaling_factor_0_is_identity() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_with_scaling_f32(&mut data, &cos_t, &sin_t, 8, 2, 0.0);
        // scale=0 → all cos/sin scaled to 0 → output is zeros
        for i in 0..8 {
            assert!(approx_eq(data[i], 0.0), "scale=0 at {i}: {}", data[i]);
        }
    }

    #[test]
    fn test_scaling_factor_applied() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut s1 = orig.clone();
        let mut s2 = orig.clone();
        rope_with_scaling_f32(&mut s1, &cos_t, &sin_t, 8, 1, 0.5);
        rope_with_scaling_f32(&mut s2, &cos_t, &sin_t, 8, 1, 1.0);
        // Different scales → different results
        assert!(s1 != s2);
    }

    #[test]
    fn test_scaling_scalar_matches_dispatcher() {
        let (cos_t, sin_t) = make_tables(16, 8, 10000.0);
        let orig: Vec<f32> = (0..16).map(|i| i as f32 * 0.15).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_with_scaling_f32(&mut disp, &cos_t, &sin_t, 16, 5, 0.7);
        scalar_rope_with_scaling_f32(&mut scal, &cos_t, &sin_t, 16, 5, 0.7);
        for i in 0..16 {
            assert!(approx_eq(disp[i], scal[i]), "scaling match at {i}");
        }
    }

    #[test]
    fn test_scaling_pos0_scale1_identity() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![3.0, -1.0, 7.0, 2.0, -5.0, 0.5, 1.0, -3.0];
        let mut data = orig.clone();
        rope_with_scaling_f32(&mut data, &cos_t, &sin_t, 8, 0, 1.0);
        for i in 0..8 {
            assert!(approx_eq(data[i], orig[i]));
        }
    }

    #[test]
    fn test_scaling_preserves_norm_with_scale1() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope_with_scaling_f32(&mut data, &cos_t, &sin_t, 8, 1, 1.0);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-3);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_head_dim_2() {
        let (cos_t, sin_t) = make_tables(2, 4, 10000.0);
        let orig = vec![1.0, 2.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 2, 1);
        let mut scalar = orig.clone();
        scalar_rope_forward_f32(&mut scalar, &cos_t, &sin_t, 2, 1);
        assert!(approx_eq(data[0], scalar[0]));
        assert!(approx_eq(data[1], scalar[1]));
    }

    #[test]
    fn test_head_dim_2_inverse() {
        let (cos_t, sin_t) = make_tables(2, 4, 10000.0);
        let orig = vec![3.0, -1.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 2, 2);
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 2, 2);
        assert!(approx_eq(data[0], orig[0]));
        assert!(approx_eq(data[1], orig[1]));
    }

    #[test]
    fn test_head_dim_2_neox() {
        let (cos_t, sin_t) = make_tables(2, 4, 10000.0);
        let orig = vec![1.0, 2.0];
        let mut data = orig.clone();
        rope_neox_f32(&mut data, &cos_t, &sin_t, 2, 1);
        let mut scalar = orig.clone();
        scalar_rope_neox_f32(&mut scalar, &cos_t, &sin_t, 2, 1);
        assert!(approx_eq(data[0], scalar[0]));
        assert!(approx_eq(data[1], scalar[1]));
    }

    #[test]
    fn test_seq_pos_0_all_ops_identity() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut fwd = orig.clone();
        rope_forward_f32(&mut fwd, &cos_t, &sin_t, 8, 0);

        let mut inv = orig.clone();
        inverse_rope_f32(&mut inv, &cos_t, &sin_t, 8, 0);

        let mut neox = orig.clone();
        rope_neox_f32(&mut neox, &cos_t, &sin_t, 8, 0);

        let mut scaled = orig.clone();
        rope_with_scaling_f32(&mut scaled, &cos_t, &sin_t, 8, 0, 1.0);

        for i in 0..8 {
            assert!(approx_eq(fwd[i], orig[i]), "fwd pos0");
            assert!(approx_eq(inv[i], orig[i]), "inv pos0");
            assert!(approx_eq(neox[i], orig[i]), "neox pos0");
            assert!(approx_eq(scaled[i], orig[i]), "scaled pos0");
        }
    }

    #[test]
    fn test_odd_half_dim_forward() {
        // head_dim=6 → half=3, which is odd → triggers scalar tail
        let (cos_t, sin_t) = make_tables(6, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_forward_f32(&mut disp, &cos_t, &sin_t, 6, 2);
        scalar_rope_forward_f32(&mut scal, &cos_t, &sin_t, 6, 2);
        for i in 0..6 {
            assert!(approx_eq(disp[i], scal[i]), "odd half at {i}");
        }
    }

    #[test]
    fn test_odd_half_dim_inverse_roundtrip() {
        let (cos_t, sin_t) = make_tables(6, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 6, 1);
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 6, 1);
        for i in 0..6 {
            assert!(approx_eq(data[i], orig[i]), "odd roundtrip at {i}");
        }
    }

    #[test]
    fn test_odd_half_dim_neox() {
        let (cos_t, sin_t) = make_tables(6, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_neox_f32(&mut disp, &cos_t, &sin_t, 6, 2);
        scalar_rope_neox_f32(&mut scal, &cos_t, &sin_t, 6, 2);
        for i in 0..6 {
            assert!(approx_eq(disp[i], scal[i]), "neox odd at {i}");
        }
    }

    #[test]
    fn test_odd_half_dim_scaling() {
        let (cos_t, sin_t) = make_tables(6, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_with_scaling_f32(&mut disp, &cos_t, &sin_t, 6, 2, 0.8);
        scalar_rope_with_scaling_f32(&mut scal, &cos_t, &sin_t, 6, 2, 0.8);
        for i in 0..6 {
            assert!(approx_eq(disp[i], scal[i]), "scaling odd at {i}");
        }
    }

    #[test]
    fn test_head_dim_4() {
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let orig = vec![1.0, 0.0, 0.0, 1.0];
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_forward_f32(&mut disp, &cos_t, &sin_t, 4, 2);
        scalar_rope_forward_f32(&mut scal, &cos_t, &sin_t, 4, 2);
        for i in 0..4 {
            assert!(approx_eq(disp[i], scal[i]));
        }
    }

    #[test]
    fn test_large_seq_pos() {
        let (cos_t, sin_t) = make_tables(8, 1024, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 1023);
        // Should not panic and should differ from identity
        assert!(data != orig);
    }

    #[test]
    fn test_all_zeros_input() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let mut data = vec![0.0f32; 8];
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 2);
        for i in 0..8 {
            assert!(approx_eq(data[i], 0.0));
        }
    }

    #[test]
    fn test_negative_values() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let mut data = orig.clone();
        rope_forward_f32(&mut data, &cos_t, &sin_t, 8, 1);
        let norm_before: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-3);
    }

    #[test]
    fn test_batch_seq_start_offset() {
        let (cos_t, sin_t) = make_tables(4, 16, 10000.0);
        let head_dim = 4;
        let num_heads = 1;
        // Single position at seq_start=10
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let mut batch = orig.clone();
        rope_batch_f32(&mut batch, &cos_t, &sin_t, num_heads, head_dim, 10, 1);
        let mut single = orig.clone();
        rope_forward_f32(&mut single, &cos_t, &sin_t, head_dim, 10);
        for i in 0..4 {
            assert!(approx_eq(batch[i], single[i]));
        }
    }

    #[test]
    fn test_inverse_not_same_as_forward() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut fwd = orig.clone();
        let mut inv = orig.clone();
        rope_forward_f32(&mut fwd, &cos_t, &sin_t, 8, 2);
        inverse_rope_f32(&mut inv, &cos_t, &sin_t, 8, 2);
        // At non-zero position, forward and inverse should differ
        assert!(fwd != inv);
    }

    #[test]
    fn test_neox_inverse_via_standard_not_possible() {
        // NeoX layout is incompatible with standard inverse
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        rope_neox_f32(&mut data, &cos_t, &sin_t, 8, 2);
        // Standard inverse won't recover NeoX-rotated data
        inverse_rope_f32(&mut data, &cos_t, &sin_t, 8, 2);
        let any_mismatch = (0..8).any(|i| !approx_eq(data[i], orig[i]));
        assert!(any_mismatch, "NeoX + standard inverse should not roundtrip");
    }

    #[test]
    fn test_forward_pair_rotation_formula() {
        // Verify the rotation formula directly
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let half = 2;
        let pos = 1;
        let offset = pos * half;
        let x0 = 3.0f32;
        let x1 = 5.0f32;
        let c = cos_t[offset + 0];
        let s = sin_t[offset + 0];
        let expected_0 = x0 * c - x1 * s;
        let expected_1 = x0 * s + x1 * c;

        let mut data = vec![x0, x1, 0.0, 0.0];
        rope_forward_f32(&mut data, &cos_t, &sin_t, 4, pos);
        assert!(approx_eq(data[0], expected_0));
        assert!(approx_eq(data[1], expected_1));
    }

    #[test]
    fn test_scaling_half_vs_full() {
        // scale=0.5 should give half the cos/sin effect
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let half = 2;
        let pos = 1;
        let offset = pos * half;
        let x0 = 2.0f32;
        let x1 = 4.0f32;
        let c = cos_t[offset + 0] * 0.5;
        let s = sin_t[offset + 0] * 0.5;
        let expected_0 = x0 * c - x1 * s;
        let expected_1 = x0 * s + x1 * c;

        let mut data = vec![x0, x1, 0.0, 0.0];
        rope_with_scaling_f32(&mut data, &cos_t, &sin_t, 4, pos, 0.5);
        assert!(approx_eq(data[0], expected_0));
        assert!(approx_eq(data[1], expected_1));
    }

    #[test]
    fn test_batch_empty_seq() {
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let orig = data.clone();
        rope_batch_f32(&mut data, &cos_t, &sin_t, 1, 4, 0, 0);
        assert_eq!(data, orig);
    }

    #[test]
    fn test_head_dim_2_batch() {
        let (cos_t, sin_t) = make_tables(2, 4, 10000.0);
        let num_heads = 4;
        let head_dim = 2;
        let orig: Vec<f32> = (0..num_heads * head_dim).map(|i| i as f32 + 1.0).collect();
        let mut batch = orig.clone();
        let mut scal = orig.clone();
        rope_batch_f32(&mut batch, &cos_t, &sin_t, num_heads, head_dim, 1, 1);
        scalar_rope_batch_f32(&mut scal, &cos_t, &sin_t, num_heads, head_dim, 1, 1);
        for i in 0..batch.len() {
            assert!(approx_eq(batch[i], scal[i]), "dim2 batch at {i}");
        }
    }

    #[test]
    fn test_head_dim_10_odd_half() {
        // head_dim=10 → half=5 (odd), tests scalar tail in all ops
        let (cos_t, sin_t) = make_tables(10, 4, 10000.0);
        let orig: Vec<f32> = (0..10).map(|i| i as f32 * 0.3).collect();

        let mut fwd_d = orig.clone();
        let mut fwd_s = orig.clone();
        rope_forward_f32(&mut fwd_d, &cos_t, &sin_t, 10, 2);
        scalar_rope_forward_f32(&mut fwd_s, &cos_t, &sin_t, 10, 2);
        for i in 0..10 {
            assert!(approx_eq(fwd_d[i], fwd_s[i]), "dim10 fwd at {i}");
        }

        let mut inv_d = fwd_d.clone();
        inverse_rope_f32(&mut inv_d, &cos_t, &sin_t, 10, 2);
        for i in 0..10 {
            assert!(approx_eq(inv_d[i], orig[i]), "dim10 roundtrip at {i}");
        }
    }

    #[test]
    fn test_table_monotonic_frequency_decay() {
        // Higher dimension indices should have lower frequency (smaller theta)
        let (_cos_t, sin_t) = make_tables(16, 4, 10000.0);
        let half = 8;
        // At pos=1, sin values should decrease in magnitude for higher dims
        // (because theta decreases → angle decreases → sin(angle) decreases)
        let s0 = sin_t[half + 0].abs();
        let s7 = sin_t[half + 7].abs();
        assert!(s0 > s7, "frequency should decay: |sin[0]|={s0} > |sin[7]|={s7}");
    }

    #[test]
    fn test_scaling_negative_scale() {
        let (cos_t, sin_t) = make_tables(8, 4, 10000.0);
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = orig.clone();
        // Negative scale should still work (just flips signs)
        rope_with_scaling_f32(&mut data, &cos_t, &sin_t, 8, 1, -1.0);
        let mut data2 = orig.clone();
        scalar_rope_with_scaling_f32(&mut data2, &cos_t, &sin_t, 8, 1, -1.0);
        for i in 0..8 {
            assert!(approx_eq(data[i], data2[i]));
        }
    }

    #[test]
    fn test_forward_consecutive_positions() {
        let (cos_t, sin_t) = make_tables(8, 8, 10000.0);
        let orig = vec![1.0, 0.5, -1.0, 0.5, 2.0, -0.5, 0.0, 1.5];
        let mut results = Vec::new();
        for pos in 0..8 {
            let mut data = orig.clone();
            rope_forward_f32(&mut data, &cos_t, &sin_t, 8, pos);
            results.push(data);
        }
        // All positions should produce different results (except pos=0 which is identity)
        for i in 1..8 {
            for j in (i + 1)..8 {
                assert!(results[i] != results[j], "pos {i} == pos {j}");
            }
        }
    }

    #[test]
    fn test_build_tables_small_base() {
        let (cos_t, sin_t) = make_tables(4, 4, 2.0);
        let half = 2;
        // With small base, angles change faster
        let theta0 = 2.0f32.powf(0.0); // = 1.0
        let angle_p1 = 1.0 * theta0; // = 1.0
        assert!(approx_eq(cos_t[half + 0], angle_p1.cos()));
        assert!(approx_eq(sin_t[half + 0], angle_p1.sin()));
    }

    #[test]
    fn test_batch_4heads_dim16_seq4() {
        let (cos_t, sin_t) = make_tables(16, 16, 10000.0);
        let num_heads = 4;
        let head_dim = 16;
        let seq_len = 4;
        let seq_start = 3;
        let total = seq_len * num_heads * head_dim;
        let orig: Vec<f32> = (0..total).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let mut disp = orig.clone();
        let mut scal = orig.clone();
        rope_batch_f32(&mut disp, &cos_t, &sin_t, num_heads, head_dim, seq_start, seq_len);
        scalar_rope_batch_f32(&mut scal, &cos_t, &sin_t, num_heads, head_dim, seq_start, seq_len);
        for i in 0..total {
            assert!(approx_eq(disp[i], scal[i]), "big batch at {i}");
        }
    }

    #[test]
    fn test_neox_dim4_specific() {
        let (cos_t, sin_t) = make_tables(4, 4, 10000.0);
        let half = 2;
        let pos = 1;
        let offset = pos * half;
        // Input: [a, b | c, d] where pairs are (a,c) and (b,d)
        let a = 1.0f32;
        let b = 2.0;
        let c = 3.0;
        let d = 4.0;
        let c0 = cos_t[offset + 0];
        let s0 = sin_t[offset + 0];
        let c1 = cos_t[offset + 1];
        let s1 = sin_t[offset + 1];
        let exp_a = a * c0 - c * s0;
        let exp_b = b * c1 - d * s1;
        let exp_c = a * s0 + c * c0;
        let exp_d = b * s1 + d * c1;

        let mut data = vec![a, b, c, d];
        rope_neox_f32(&mut data, &cos_t, &sin_t, 4, pos);
        assert!(approx_eq(data[0], exp_a), "neox a");
        assert!(approx_eq(data[1], exp_b), "neox b");
        assert!(approx_eq(data[2], exp_c), "neox c");
        assert!(approx_eq(data[3], exp_d), "neox d");
    }
}
