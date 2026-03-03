//! ARM NEON-optimized data layout transformations for Apple Silicon.
//!
//! Provides AoS↔SoA conversion, multi-channel interleave/deinterleave,
//! 4×4 f32 matrix transpose via NEON `vtrn1q_f32`/`vtrn2q_f32` and
//! `vzip1q_f32`/`vzip2q_f32`, 2-bit/4-bit pack/unpack for quantised data,
//! padding/alignment utilities, and NCHW↔NHWC layout conversion.
//!
//! All hot paths use `float32x4_t` NEON intrinsics with scalar fallback
//! for remainder elements.

use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── AoS ↔ SoA ──────────────────────────────────────────────────────────

/// Convert AoS (Array of Structs) layout to SoA (Struct of Arrays) for
/// `channels` interleaved channels of `f32`.
///
/// `input` has length `count * channels` laid out as
/// `[a0, b0, …, a1, b1, …]`. Each output slice in `outputs` receives the
/// values for that channel.
///
/// # Panics
///
/// Panics if `outputs.len() != channels`, any output slice is shorter than
/// `count`, or `input.len() < count * channels`.
pub fn aos_to_soa(input: &[f32], channels: usize, count: usize, outputs: &mut [&mut [f32]]) {
    assert_eq!(outputs.len(), channels, "outputs.len() must equal channels");
    assert!(
        input.len() >= count * channels,
        "input too short: {} < {}",
        input.len(),
        count * channels
    );
    for out in outputs.iter() {
        assert!(out.len() >= count, "output slice too short");
    }
    for i in 0..count {
        for ch in 0..channels {
            outputs[ch][i] = input[i * channels + ch];
        }
    }
}

/// Convert SoA (Struct of Arrays) back to AoS (Array of Structs).
///
/// `inputs` holds one slice per channel; `output` is filled in interleaved
/// order.
///
/// # Panics
///
/// Panics if `inputs.len() != channels`, any input slice is shorter than
/// `count`, or `output.len() < count * channels`.
pub fn soa_to_aos(inputs: &[&[f32]], channels: usize, count: usize, output: &mut [f32]) {
    assert_eq!(inputs.len(), channels, "inputs.len() must equal channels");
    assert!(
        output.len() >= count * channels,
        "output too short: {} < {}",
        output.len(),
        count * channels
    );
    for inp in inputs.iter() {
        assert!(inp.len() >= count, "input slice too short");
    }
    for i in 0..count {
        for ch in 0..channels {
            output[i * channels + ch] = inputs[ch][i];
        }
    }
}

// ── Interleave / Deinterleave (2-channel, NEON fast path) ───────────

/// Interleave two f32 channels using NEON `vzip1q_f32` / `vzip2q_f32`.
///
/// Produces `[a0, b0, a1, b1, …]` from separate `a` and `b` arrays.
///
/// # Panics
///
/// Panics if `a.len() < count`, `b.len() < count`, or
/// `output.len() < count * 2`.
pub fn interleave_2ch(a: &[f32], b: &[f32], count: usize, output: &mut [f32]) {
    assert!(a.len() >= count);
    assert!(b.len() >= count);
    assert!(output.len() >= count * 2);

    let simd_end = count / LANES * LANES;
    let mut o = 0usize;

    // SAFETY: pointers are valid for the asserted lengths; NEON is
    // guaranteed on aarch64.
    unsafe {
        let mut i = 0usize;
        while i < simd_end {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let lo = vzip1q_f32(va, vb);
            let hi = vzip2q_f32(va, vb);
            vst1q_f32(output.as_mut_ptr().add(o), lo);
            vst1q_f32(output.as_mut_ptr().add(o + LANES), hi);
            i += LANES;
            o += LANES * 2;
        }
        // scalar remainder
        for j in simd_end..count {
            output[o] = a[j];
            output[o + 1] = b[j];
            o += 2;
        }
    }
}

/// Deinterleave a 2-channel interleaved stream back into separate arrays
/// using NEON `vuzp1q_f32` / `vuzp2q_f32`.
///
/// # Panics
///
/// Panics if `input.len() < count * 2`, `a.len() < count`, or
/// `b.len() < count`.
pub fn deinterleave_2ch(input: &[f32], count: usize, a: &mut [f32], b: &mut [f32]) {
    assert!(input.len() >= count * 2);
    assert!(a.len() >= count);
    assert!(b.len() >= count);

    let simd_end = count / LANES * LANES;

    unsafe {
        let mut s = 0usize;
        let mut d = 0usize;
        while d < simd_end {
            let lo = vld1q_f32(input.as_ptr().add(s));
            let hi = vld1q_f32(input.as_ptr().add(s + LANES));
            let va = vuzp1q_f32(lo, hi);
            let vb = vuzp2q_f32(lo, hi);
            vst1q_f32(a.as_mut_ptr().add(d), va);
            vst1q_f32(b.as_mut_ptr().add(d), vb);
            s += LANES * 2;
            d += LANES;
        }
        // scalar remainder
        for j in simd_end..count {
            a[j] = input[j * 2];
            b[j] = input[j * 2 + 1];
        }
    }
}

/// Interleave four f32 channels using NEON.
///
/// Produces `[a0, b0, c0, d0, a1, b1, c1, d1, …]`.
///
/// # Panics
///
/// Panics if any input slice is shorter than `count` or `output` is
/// shorter than `count * 4`.
pub fn interleave_4ch(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    count: usize,
    output: &mut [f32],
) {
    assert!(a.len() >= count);
    assert!(b.len() >= count);
    assert!(c.len() >= count);
    assert!(d.len() >= count);
    assert!(output.len() >= count * 4);

    unsafe {
        let simd_end = count / LANES * LANES;
        let mut o = 0usize;
        let mut i = 0usize;
        while i < simd_end {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vld1q_f32(c.as_ptr().add(i));
            let vd = vld1q_f32(d.as_ptr().add(i));
            // Transpose 4×4 to get interleaved columns.
            let ab_lo = vzip1q_f32(va, vb);
            let ab_hi = vzip2q_f32(va, vb);
            let cd_lo = vzip1q_f32(vc, vd);
            let cd_hi = vzip2q_f32(vc, vd);

            // Combine pairs: cast to 64-bit to merge two f32 lanes at a
            // time, giving the final interleaved order.
            let r0 = vcombine_f32(vget_low_f32(ab_lo), vget_low_f32(cd_lo));
            let r1 = vcombine_f32(vget_high_f32(ab_lo), vget_high_f32(cd_lo));
            let r2 = vcombine_f32(vget_low_f32(ab_hi), vget_low_f32(cd_hi));
            let r3 = vcombine_f32(vget_high_f32(ab_hi), vget_high_f32(cd_hi));
            vst1q_f32(output.as_mut_ptr().add(o), r0);
            vst1q_f32(output.as_mut_ptr().add(o + 4), r1);
            vst1q_f32(output.as_mut_ptr().add(o + 8), r2);
            vst1q_f32(output.as_mut_ptr().add(o + 12), r3);
            i += LANES;
            o += LANES * 4;
        }
        for j in simd_end..count {
            output[j * 4] = a[j];
            output[j * 4 + 1] = b[j];
            output[j * 4 + 2] = c[j];
            output[j * 4 + 3] = d[j];
        }
    }
}

/// Deinterleave a 4-channel stream into separate arrays.
///
/// # Panics
///
/// Panics if `input.len() < count * 4` or any output is shorter than
/// `count`.
pub fn deinterleave_4ch(
    input: &[f32],
    count: usize,
    a: &mut [f32],
    b: &mut [f32],
    c: &mut [f32],
    d: &mut [f32],
) {
    assert!(input.len() >= count * 4);
    assert!(a.len() >= count);
    assert!(b.len() >= count);
    assert!(c.len() >= count);
    assert!(d.len() >= count);

    for j in 0..count {
        a[j] = input[j * 4];
        b[j] = input[j * 4 + 1];
        c[j] = input[j * 4 + 2];
        d[j] = input[j * 4 + 3];
    }
}

// ── 4×4 f32 transpose via NEON vtrn ────────────────────────────────

/// Transpose a 4×4 f32 block in-place using NEON `vtrn1q_f32` /
/// `vtrn2q_f32` combined with `vzip1q_f32` / `vzip2q_f32`.
///
/// `data` must contain exactly 16 elements laid out in row-major order.
///
/// # Panics
///
/// Panics if `data.len() < 16`.
pub fn transpose_4x4_inplace(data: &mut [f32]) {
    assert!(data.len() >= 16, "need at least 16 f32 elements");
    unsafe {
        let r0 = vld1q_f32(data.as_ptr());
        let r1 = vld1q_f32(data.as_ptr().add(4));
        let r2 = vld1q_f32(data.as_ptr().add(8));
        let r3 = vld1q_f32(data.as_ptr().add(12));

        // Stage 1: transpose 2×2 blocks of f32 pairs.
        let t0 = vtrn1q_f32(r0, r1);
        let t1 = vtrn2q_f32(r0, r1);
        let t2 = vtrn1q_f32(r2, r3);
        let t3 = vtrn2q_f32(r2, r3);

        // Stage 2: swap 64-bit halves to complete the full 4×4 transpose.
        // Reinterpret as f64 pairs to exchange the 64-bit high/low halves.
        let u0 = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
        let u1 = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
        let u2 = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
        let u3 = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));

        vst1q_f32(data.as_mut_ptr(), u0);
        vst1q_f32(data.as_mut_ptr().add(4), u1);
        vst1q_f32(data.as_mut_ptr().add(8), u2);
        vst1q_f32(data.as_mut_ptr().add(12), u3);
    }
}

/// Transpose a 4×4 f32 block from `input` into `output`.
///
/// Both slices must hold at least 16 elements.
pub fn transpose_4x4(input: &[f32], output: &mut [f32]) {
    assert!(input.len() >= 16);
    assert!(output.len() >= 16);
    output[..16].copy_from_slice(&input[..16]);
    transpose_4x4_inplace(output);
}

// ── 2-bit pack / unpack ─────────────────────────────────────────────

/// Pack an array of 2-bit signed values (-1, 0, 1 stored as `i8`) into a
/// byte stream, four values per byte (LSB first).
///
/// The two-bit encoding is: `val & 0x03` (i.e. the lowest two bits of
/// each `i8`).
///
/// # Panics
///
/// Panics if `output.len() < ceil(values.len() / 4)`.
pub fn pack_2bit(values: &[i8], output: &mut [u8]) {
    let packed_len = (values.len() + 3) / 4;
    assert!(output.len() >= packed_len, "output too short for 2-bit packing");
    for (i, chunk) in values.chunks(4).enumerate() {
        let mut byte: u8 = 0;
        for (j, &v) in chunk.iter().enumerate() {
            byte |= ((v as u8) & 0x03) << (j * 2);
        }
        output[i] = byte;
    }
}

/// Unpack a 2-bit packed byte stream back into `i8` values (sign-extended
/// from 2-bit two's complement: 0b11 → -1, 0b10 → -2, 0b01 → 1, 0b00 → 0).
///
/// # Panics
///
/// Panics if `output.len() < count` or `input` has fewer bytes than
/// needed.
pub fn unpack_2bit(input: &[u8], count: usize, output: &mut [i8]) {
    let needed_bytes = (count + 3) / 4;
    assert!(input.len() >= needed_bytes, "input too short for 2-bit unpack");
    assert!(output.len() >= count, "output too short");
    let mut idx = 0usize;
    for &byte in &input[..needed_bytes] {
        for shift in 0..4 {
            if idx >= count {
                break;
            }
            let bits = (byte >> (shift * 2)) & 0x03;
            // Sign-extend from 2 bits.
            output[idx] = ((bits as i8) << 6) >> 6;
            idx += 1;
        }
    }
}

// ── 4-bit pack / unpack ─────────────────────────────────────────────

/// Pack an array of 4-bit values (0..15 stored as `u8`) into a byte
/// stream, two values per byte (low nibble first).
///
/// # Panics
///
/// Panics if `output.len() < ceil(values.len() / 2)`.
pub fn pack_4bit(values: &[u8], output: &mut [u8]) {
    let packed_len = (values.len() + 1) / 2;
    assert!(output.len() >= packed_len, "output too short for 4-bit packing");
    for (i, chunk) in values.chunks(2).enumerate() {
        let lo = chunk[0] & 0x0F;
        let hi = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
        output[i] = lo | (hi << 4);
    }
}

/// Unpack a 4-bit packed byte stream back into `u8` nibbles.
///
/// # Panics
///
/// Panics if `output.len() < count` or `input` has fewer bytes than
/// needed.
pub fn unpack_4bit(input: &[u8], count: usize, output: &mut [u8]) {
    let needed_bytes = (count + 1) / 2;
    assert!(input.len() >= needed_bytes, "input too short for 4-bit unpack");
    assert!(output.len() >= count, "output too short");
    let mut idx = 0usize;
    for &byte in &input[..needed_bytes] {
        if idx < count {
            output[idx] = byte & 0x0F;
            idx += 1;
        }
        if idx < count {
            output[idx] = (byte >> 4) & 0x0F;
            idx += 1;
        }
    }
}

// ── Alignment / Padding Utilities ───────────────────────────────────

/// Round `n` up to the next multiple of `LANES` (4) for NEON-friendly
/// buffer allocation.
#[inline]
pub const fn align_to_neon(n: usize) -> usize {
    (n + LANES - 1) & !(LANES - 1)
}

/// Allocate a zero-initialised `Vec<f32>` whose length is rounded up to
/// a multiple of 4 (NEON lane width).
pub fn alloc_aligned_f32(min_len: usize) -> Vec<f32> {
    vec![0.0f32; align_to_neon(min_len)]
}

/// Pad `input` with trailing zeros so the returned vector length is a
/// multiple of `LANES`.
pub fn pad_to_neon(input: &[f32]) -> Vec<f32> {
    let aligned = align_to_neon(input.len());
    let mut out = vec![0.0f32; aligned];
    out[..input.len()].copy_from_slice(input);
    out
}

/// Return `true` if `n` is a multiple of `LANES`.
#[inline]
pub const fn is_neon_aligned(n: usize) -> bool {
    n.is_multiple_of(LANES)
}

// ── NCHW ↔ NHWC Layout Conversion ──────────────────────────────────

/// Convert NCHW (channel-first) layout to NHWC (channel-last).
///
/// `input` has shape `[N, C, H, W]` in row-major order; `output` receives
/// the same data in `[N, H, W, C]` order.
///
/// When `C == 4` a NEON fast path interleaves four planar channels
/// directly. Otherwise a scalar loop is used.
///
/// # Panics
///
/// Panics if slice lengths do not match `n * c * h * w`.
pub fn nchw_to_nhwc(input: &[f32], n: usize, c: usize, h: usize, w: usize, output: &mut [f32]) {
    let total = n * c * h * w;
    assert!(input.len() >= total, "input too short");
    assert!(output.len() >= total, "output too short");

    let hw = h * w;
    for batch in 0..n {
        let in_base = batch * c * hw;
        let out_base = batch * hw * c;
        if c == 4 {
            // NEON fast path for 4 channels.
            let a = &input[in_base..in_base + hw];
            let b = &input[in_base + hw..in_base + 2 * hw];
            let ch_c = &input[in_base + 2 * hw..in_base + 3 * hw];
            let d = &input[in_base + 3 * hw..in_base + 4 * hw];
            interleave_4ch(a, b, ch_c, d, hw, &mut output[out_base..out_base + hw * 4]);
        } else {
            for pixel in 0..hw {
                for ch in 0..c {
                    output[out_base + pixel * c + ch] = input[in_base + ch * hw + pixel];
                }
            }
        }
    }
}

/// Convert NHWC (channel-last) layout to NCHW (channel-first).
///
/// Inverse of [`nchw_to_nhwc`].
///
/// # Panics
///
/// Panics if slice lengths do not match `n * c * h * w`.
pub fn nhwc_to_nchw(input: &[f32], n: usize, c: usize, h: usize, w: usize, output: &mut [f32]) {
    let total = n * c * h * w;
    assert!(input.len() >= total, "input too short");
    assert!(output.len() >= total, "output too short");

    let hw = h * w;
    for batch in 0..n {
        let in_base = batch * hw * c;
        let out_base = batch * c * hw;
        if c == 4 {
            let mut a = vec![0.0f32; hw];
            let mut b = vec![0.0f32; hw];
            let mut ch_c = vec![0.0f32; hw];
            let mut d = vec![0.0f32; hw];
            deinterleave_4ch(
                &input[in_base..in_base + hw * 4],
                hw,
                &mut a,
                &mut b,
                &mut ch_c,
                &mut d,
            );
            output[out_base..out_base + hw].copy_from_slice(&a);
            output[out_base + hw..out_base + 2 * hw].copy_from_slice(&b);
            output[out_base + 2 * hw..out_base + 3 * hw].copy_from_slice(&ch_c);
            output[out_base + 3 * hw..out_base + 4 * hw].copy_from_slice(&d);
        } else {
            for pixel in 0..hw {
                for ch in 0..c {
                    output[out_base + ch * hw + pixel] = input[in_base + pixel * c + ch];
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AoS ↔ SoA ──────────────────────────────────────────────────

    #[test]
    fn test_aos_to_soa_2ch() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut ch0 = [0.0f32; 3];
        let mut ch1 = [0.0f32; 3];
        aos_to_soa(&input, 2, 3, &mut [&mut ch0, &mut ch1]);
        assert_eq!(ch0, [1.0, 3.0, 5.0]);
        assert_eq!(ch1, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_soa_to_aos_2ch() {
        let ch0: [f32; 3] = [1.0, 3.0, 5.0];
        let ch1: [f32; 3] = [2.0, 4.0, 6.0];
        let mut out = [0.0f32; 6];
        soa_to_aos(&[&ch0[..], &ch1[..]], 2, 3, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_aos_soa_roundtrip_3ch() {
        let input: Vec<f32> = (0..30).map(|x| x as f32).collect();
        let count = 10;
        let channels = 3;
        let mut c0 = vec![0.0f32; count];
        let mut c1 = vec![0.0f32; count];
        let mut c2 = vec![0.0f32; count];
        aos_to_soa(&input, channels, count, &mut [&mut c0, &mut c1, &mut c2]);
        let mut roundtrip = vec![0.0f32; 30];
        soa_to_aos(&[&c0, &c1, &c2], channels, count, &mut roundtrip);
        assert_eq!(input, roundtrip);
    }

    #[test]
    fn test_aos_to_soa_single_element() {
        let input = [42.0f32, 99.0];
        let mut ch0 = [0.0f32; 1];
        let mut ch1 = [0.0f32; 1];
        aos_to_soa(&input, 2, 1, &mut [&mut ch0, &mut ch1]);
        assert_eq!(ch0, [42.0]);
        assert_eq!(ch1, [99.0]);
    }

    // ── Interleave / Deinterleave 2-channel ─────────────────────────

    #[test]
    fn test_interleave_2ch_small() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        let mut out = [0.0f32; 6];
        interleave_2ch(&a, &b, 3, &mut out);
        assert_eq!(out, [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    }

    #[test]
    fn test_deinterleave_2ch_small() {
        let input = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        deinterleave_2ch(&input, 3, &mut a, &mut b);
        assert_eq!(a, [1.0, 2.0, 3.0]);
        assert_eq!(b, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_interleave_deinterleave_2ch_roundtrip() {
        let a: Vec<f32> = (0..17).map(|x| x as f32).collect();
        let b: Vec<f32> = (100..117).map(|x| x as f32).collect();
        let mut interleaved = vec![0.0f32; 34];
        interleave_2ch(&a, &b, 17, &mut interleaved);
        let mut ra = vec![0.0f32; 17];
        let mut rb = vec![0.0f32; 17];
        deinterleave_2ch(&interleaved, 17, &mut ra, &mut rb);
        assert_eq!(a, ra);
        assert_eq!(b, rb);
    }

    #[test]
    fn test_interleave_2ch_exact_lanes() {
        // Exactly 4 elements — hits NEON path with no remainder.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 8];
        interleave_2ch(&a, &b, 4, &mut out);
        assert_eq!(out, [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn test_interleave_2ch_large() {
        let n = 256;
        let a: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..n).map(|x| (x as f32) * 0.5).collect();
        let mut out = vec![0.0f32; n * 2];
        interleave_2ch(&a, &b, n, &mut out);
        for i in 0..n {
            assert_eq!(out[i * 2], a[i]);
            assert_eq!(out[i * 2 + 1], b[i]);
        }
    }

    // ── Interleave / Deinterleave 4-channel ─────────────────────────

    #[test]
    fn test_interleave_4ch_small() {
        let a = [1.0f32];
        let b = [2.0f32];
        let c = [3.0f32];
        let d = [4.0f32];
        let mut out = [0.0f32; 4];
        interleave_4ch(&a, &b, &c, &d, 1, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_deinterleave_4ch_small() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut a = [0.0f32; 2];
        let mut b = [0.0f32; 2];
        let mut c = [0.0f32; 2];
        let mut d = [0.0f32; 2];
        deinterleave_4ch(&input, 2, &mut a, &mut b, &mut c, &mut d);
        assert_eq!(a, [1.0, 5.0]);
        assert_eq!(b, [2.0, 6.0]);
        assert_eq!(c, [3.0, 7.0]);
        assert_eq!(d, [4.0, 8.0]);
    }

    #[test]
    fn test_interleave_deinterleave_4ch_roundtrip() {
        let n = 9;
        let a: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let b: Vec<f32> = (10..10 + n as i32).map(|x| x as f32).collect();
        let c: Vec<f32> = (20..20 + n as i32).map(|x| x as f32).collect();
        let d: Vec<f32> = (30..30 + n as i32).map(|x| x as f32).collect();
        let mut interleaved = vec![0.0f32; n * 4];
        interleave_4ch(&a, &b, &c, &d, n, &mut interleaved);
        let mut ra = vec![0.0f32; n];
        let mut rb = vec![0.0f32; n];
        let mut rc = vec![0.0f32; n];
        let mut rd = vec![0.0f32; n];
        deinterleave_4ch(&interleaved, n, &mut ra, &mut rb, &mut rc, &mut rd);
        assert_eq!(a, ra);
        assert_eq!(b, rb);
        assert_eq!(c, rc);
        assert_eq!(d, rd);
    }

    // ── 4×4 transpose ───────────────────────────────────────────────

    #[test]
    fn test_transpose_4x4_identity() {
        #[rustfmt::skip]
        let input = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut out = [0.0f32; 16];
        transpose_4x4(&input, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn test_transpose_4x4_sequential() {
        #[rustfmt::skip]
        let input = [
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0,  10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut out = [0.0f32; 16];
        transpose_4x4(&input, &mut out);
        #[rustfmt::skip]
        let expected = [
            1.0, 5.0, 9.0,  13.0,
            2.0, 6.0, 10.0, 14.0,
            3.0, 7.0, 11.0, 15.0,
            4.0, 8.0, 12.0, 16.0,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_transpose_4x4_inplace_roundtrip() {
        #[rustfmt::skip]
        let mut data = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let original = data;
        transpose_4x4_inplace(&mut data);
        transpose_4x4_inplace(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_transpose_4x4_negative_values() {
        #[rustfmt::skip]
        let mut data = [
            -1.0,  2.0, -3.0,  4.0,
             5.0, -6.0,  7.0, -8.0,
            -9.0, 10.0, -11.0, 12.0,
            13.0, -14.0, 15.0, -16.0,
        ];
        transpose_4x4_inplace(&mut data);
        // Check a few elements to ensure correctness.
        assert_eq!(data[1], 5.0); // (0,1) was (1,0)
        assert_eq!(data[4], 2.0); // (1,0) was (0,1)
    }

    // ── 2-bit pack / unpack ─────────────────────────────────────────

    #[test]
    fn test_pack_unpack_2bit_basic() {
        let values: Vec<i8> = vec![0, 1, -1, 0];
        let mut packed = vec![0u8; 1];
        pack_2bit(&values, &mut packed);
        let mut unpacked = vec![0i8; 4];
        unpack_2bit(&packed, 4, &mut unpacked);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_2bit_all_ones() {
        let values = vec![1i8; 8];
        let mut packed = vec![0u8; 2];
        pack_2bit(&values, &mut packed);
        // Each byte: 0b01_01_01_01 = 0x55
        assert_eq!(packed, [0x55, 0x55]);
    }

    #[test]
    fn test_pack_unpack_2bit_roundtrip_partial() {
        // Non-multiple-of-4 length.
        let values: Vec<i8> = vec![1, 0, -1, 1, 0];
        let packed_len = (values.len() + 3) / 4;
        let mut packed = vec![0u8; packed_len];
        pack_2bit(&values, &mut packed);
        let mut unpacked = vec![0i8; 5];
        unpack_2bit(&packed, 5, &mut unpacked);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_unpack_2bit_sign_extension() {
        // 0b11 should unpack to -1.
        let packed = [0b11_00_01_11u8]; // values: -1, 1, 0, -1
        let mut out = [0i8; 4];
        unpack_2bit(&packed, 4, &mut out);
        assert_eq!(out, [-1, 1, 0, -1]);
    }

    // ── 4-bit pack / unpack ─────────────────────────────────────────

    #[test]
    fn test_pack_unpack_4bit_basic() {
        let values: Vec<u8> = vec![0, 15, 7, 8];
        let mut packed = vec![0u8; 2];
        pack_4bit(&values, &mut packed);
        let mut unpacked = vec![0u8; 4];
        unpack_4bit(&packed, 4, &mut unpacked);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_unpack_4bit_odd_count() {
        let values: Vec<u8> = vec![3, 12, 5];
        let packed_len = (values.len() + 1) / 2;
        let mut packed = vec![0u8; packed_len];
        pack_4bit(&values, &mut packed);
        let mut unpacked = vec![0u8; 3];
        unpack_4bit(&packed, 3, &mut unpacked);
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_pack_4bit_encoding() {
        let values = [0xA_u8, 0xB];
        let mut packed = [0u8; 1];
        pack_4bit(&values, &mut packed);
        // low nibble = A, high nibble = B → 0xBA
        assert_eq!(packed[0], 0xBA);
    }

    // ── Alignment / Padding ─────────────────────────────────────────

    #[test]
    fn test_align_to_neon() {
        assert_eq!(align_to_neon(0), 0);
        assert_eq!(align_to_neon(1), 4);
        assert_eq!(align_to_neon(4), 4);
        assert_eq!(align_to_neon(5), 8);
        assert_eq!(align_to_neon(16), 16);
        assert_eq!(align_to_neon(17), 20);
    }

    #[test]
    fn test_is_neon_aligned() {
        assert!(is_neon_aligned(0));
        assert!(is_neon_aligned(4));
        assert!(is_neon_aligned(16));
        assert!(!is_neon_aligned(1));
        assert!(!is_neon_aligned(7));
    }

    #[test]
    fn test_pad_to_neon() {
        let v = pad_to_neon(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 4);
        assert_eq!(&v[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(v[3], 0.0);
    }

    #[test]
    fn test_alloc_aligned_f32() {
        let v = alloc_aligned_f32(5);
        assert_eq!(v.len(), 8);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_pad_to_neon_already_aligned() {
        let v = pad_to_neon(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.len(), 4);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── NCHW ↔ NHWC ────────────────────────────────────────────────

    #[test]
    fn test_nchw_to_nhwc_simple() {
        // N=1, C=2, H=2, W=2
        #[rustfmt::skip]
        let nchw = [
            // channel 0
            1.0, 2.0,
            3.0, 4.0,
            // channel 1
            5.0, 6.0,
            7.0, 8.0,
        ];
        let mut nhwc = [0.0f32; 8];
        nchw_to_nhwc(&nchw, 1, 2, 2, 2, &mut nhwc);
        assert_eq!(nhwc, [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn test_nhwc_to_nchw_simple() {
        let nhwc = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
        let mut nchw = [0.0f32; 8];
        nhwc_to_nchw(&nhwc, 1, 2, 2, 2, &mut nchw);
        #[rustfmt::skip]
        let expected = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        assert_eq!(nchw, expected);
    }

    #[test]
    fn test_nchw_nhwc_roundtrip() {
        let n = 2;
        let c = 3;
        let h = 4;
        let w = 5;
        let total = n * c * h * w;
        let nchw: Vec<f32> = (0..total).map(|x| x as f32).collect();
        let mut nhwc = vec![0.0f32; total];
        let mut roundtrip = vec![0.0f32; total];
        nchw_to_nhwc(&nchw, n, c, h, w, &mut nhwc);
        nhwc_to_nchw(&nhwc, n, c, h, w, &mut roundtrip);
        assert_eq!(nchw, roundtrip);
    }

    #[test]
    fn test_nchw_to_nhwc_4ch_fast_path() {
        // C=4 triggers the NEON interleave fast path.
        let n = 1;
        let c = 4;
        let h = 2;
        let w = 3;
        let hw = h * w;
        let total = n * c * hw;
        let nchw: Vec<f32> = (0..total).map(|x| x as f32).collect();
        let mut nhwc = vec![0.0f32; total];
        nchw_to_nhwc(&nchw, n, c, h, w, &mut nhwc);
        // Verify pixel (0,0): channels 0..3 should be at positions 0,6,12,18
        // in NCHW, and at positions 0..3 in NHWC.
        for ch in 0..c {
            assert_eq!(nhwc[ch], nchw[ch * hw]);
        }
    }

    #[test]
    fn test_nhwc_to_nchw_4ch_fast_path() {
        let n = 1;
        let c = 4;
        let h = 3;
        let w = 2;
        let total = n * c * h * w;
        let nchw_orig: Vec<f32> = (0..total).map(|x| x as f32).collect();
        let mut nhwc = vec![0.0f32; total];
        nchw_to_nhwc(&nchw_orig, n, c, h, w, &mut nhwc);
        let mut nchw_back = vec![0.0f32; total];
        nhwc_to_nchw(&nhwc, n, c, h, w, &mut nchw_back);
        assert_eq!(nchw_orig, nchw_back);
    }

    #[test]
    fn test_nchw_nhwc_single_pixel() {
        // N=1, C=3, H=1, W=1  →  one pixel with 3 channels.
        let nchw = [10.0, 20.0, 30.0];
        let mut nhwc = [0.0f32; 3];
        nchw_to_nhwc(&nchw, 1, 3, 1, 1, &mut nhwc);
        assert_eq!(nhwc, [10.0, 20.0, 30.0]);
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_aos_to_soa_empty() {
        let input: [f32; 0] = [];
        aos_to_soa(&input, 2, 0, &mut [&mut [][..], &mut [][..]]);
    }

    #[test]
    fn test_interleave_2ch_empty() {
        let mut out = [0.0f32; 0];
        interleave_2ch(&[], &[], 0, &mut out);
    }

    #[test]
    fn test_pack_unpack_2bit_empty() {
        let mut packed = [0u8; 0];
        pack_2bit(&[], &mut packed);
        let mut unpacked = [0i8; 0];
        unpack_2bit(&[], 0, &mut unpacked);
    }
}
