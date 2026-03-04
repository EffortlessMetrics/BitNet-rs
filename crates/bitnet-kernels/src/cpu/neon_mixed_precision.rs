//! NEON-optimized mixed-precision arithmetic for Apple Silicon.
//!
//! Provides ARM NEON SIMD-accelerated conversions between f16, bf16,
//! and f32 formats, plus mixed-precision dot products and
//! accumulation for LayerNorm weights and output quantization.
//!
//! Each function uses NEON intrinsics on `aarch64` with a scalar
//! fallback on other architectures. The NEON f16 conversion
//! intrinsics (`stdarch_neon_f16`) are not yet stable, so f16↔f32
//! conversion is done via `half::f16` methods with NEON used for
//! vectorised f32 loads, stores, and arithmetic. bf16↔f32 uses
//! full NEON integer bit-shifting.

use half::f16;

// ── helpers ────────────────────────────────────────────────────────

/// Round-to-nearest-even conversion from f32 bits to bf16 bits.
#[inline]
fn f32_bits_to_bf16(bits: u32) -> u16 {
    let lsb = (bits >> 16) & 1;
    let rounding = 0x7FFF_u32.wrapping_add(lsb);
    (bits.wrapping_add(rounding) >> 16) as u16
}

/// Convert bf16 bits to f32.
#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ── f16 → f32 ──────────────────────────────────────────────────────

/// Convert a slice of `f16` values to `f32`.
///
/// On aarch64 the NEON path batches four scalar conversions into a
/// single 128-bit store. Scalar fallback elsewhere.
pub fn f16_to_f32_neon(input: &[f16]) -> Vec<f32> {
    let len = input.len();
    let mut out = vec![0.0f32; len];

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut offset = 0;
        for chunk in input.chunks_exact(4) {
            let buf = [chunk[0].to_f32(), chunk[1].to_f32(), chunk[2].to_f32(), chunk[3].to_f32()];
            unsafe {
                let v = vld1q_f32(buf.as_ptr());
                vst1q_f32(out.as_mut_ptr().add(offset), v);
            }
            offset += 4;
        }
        for (o, v) in out[offset..].iter_mut().zip(input[offset..].iter()) {
            *o = v.to_f32();
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = len;
        for (o, v) in out.iter_mut().zip(input.iter()) {
            *o = v.to_f32();
        }
    }

    out
}

// ── f32 → f16 ──────────────────────────────────────────────────────

/// Convert a slice of `f32` values to `f16`.
///
/// On aarch64 the NEON path batches four values via a 128-bit load
/// before scalar f16 conversion. Scalar fallback elsewhere.
pub fn f32_to_f16_neon(input: &[f32]) -> Vec<f16> {
    let len = input.len();
    let mut out = vec![f16::ZERO; len];

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut offset = 0;
        for chunk in input.chunks_exact(4) {
            let mut buf = [0.0f32; 4];
            unsafe {
                let v = vld1q_f32(chunk.as_ptr());
                vst1q_f32(buf.as_mut_ptr(), v);
            }
            for (o, &v) in out[offset..offset + 4].iter_mut().zip(buf.iter()) {
                *o = f16::from_f32(v);
            }
            offset += 4;
        }
        for (o, &v) in out[offset..].iter_mut().zip(input[offset..].iter()) {
            *o = f16::from_f32(v);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = len;
        for (o, &v) in out.iter_mut().zip(input.iter()) {
            *o = f16::from_f32(v);
        }
    }

    out
}

// ── bf16 → f32 ─────────────────────────────────────────────────────

/// Convert bf16 values (raw `u16` bits) to `f32`.
///
/// Full NEON path on aarch64: widen u16→u32, shift left 16,
/// reinterpret as f32. Scalar bit-shift elsewhere.
pub fn bf16_to_f32_neon(input: &[u16]) -> Vec<f32> {
    let len = input.len();
    let mut out = vec![0.0f32; len];

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut offset = 0;
        for chunk in input.chunks_exact(4) {
            unsafe {
                let bits = vld1_u16(chunk.as_ptr());
                let wide = vmovl_u16(bits);
                let shifted = vshlq_n_u32::<16>(wide);
                let fv = vreinterpretq_f32_u32(shifted);
                vst1q_f32(out.as_mut_ptr().add(offset), fv);
            }
            offset += 4;
        }
        for (o, &b) in out[offset..].iter_mut().zip(input[offset..].iter()) {
            *o = bf16_bits_to_f32(b);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = len;
        for (o, &b) in out.iter_mut().zip(input.iter()) {
            *o = bf16_bits_to_f32(b);
        }
    }

    out
}

// ── f32 → bf16 ─────────────────────────────────────────────────────

/// Convert `f32` values to bf16 (returned as raw `u16` bits).
///
/// Uses round-to-nearest-even. Full NEON bit manipulation on
/// aarch64; scalar path elsewhere.
pub fn f32_to_bf16_neon(input: &[f32]) -> Vec<u16> {
    let len = input.len();
    let mut out = vec![0u16; len];

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut offset = 0;
        for chunk in input.chunks_exact(4) {
            unsafe {
                let fv = vld1q_f32(chunk.as_ptr());
                let bits = vreinterpretq_u32_f32(fv);
                let shifted = vshrq_n_u32::<16>(bits);
                let one = vdupq_n_u32(1);
                let lsb = vandq_u32(shifted, one);
                let base = vdupq_n_u32(0x7FFF);
                let bias = vaddq_u32(base, lsb);
                let rounded = vaddq_u32(bits, bias);
                let upper = vshrq_n_u32::<16>(rounded);
                let narrow = vmovn_u32(upper);
                vst1_u16(out.as_mut_ptr().add(offset), narrow);
            }
            offset += 4;
        }
        for (o, &v) in out[offset..].iter_mut().zip(input[offset..].iter()) {
            *o = f32_bits_to_bf16(v.to_bits());
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = len;
        for (o, &v) in out.iter_mut().zip(input.iter()) {
            *o = f32_bits_to_bf16(v.to_bits());
        }
    }

    out
}

// ── mixed dot product ──────────────────────────────────────────────

/// Dot product of an `f16` vector with an `f32` vector.
///
/// Converts `f16` elements to `f32` then uses NEON multiply-
/// accumulate on aarch64. Scalar multiply-add elsewhere.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn mixed_dot_f16_f32(a: &[f16], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "mixed_dot: length mismatch {} vs {}", a.len(), b.len(),);
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut offset = 0;
        for (ac, bc) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let buf = [ac[0].to_f32(), ac[1].to_f32(), ac[2].to_f32(), ac[3].to_f32()];
            unsafe {
                let a_fv = vld1q_f32(buf.as_ptr());
                let b_fv = vld1q_f32(bc.as_ptr());
                acc = vmlaq_f32(acc, a_fv, b_fv);
            }
            offset += 4;
        }
        let mut result = unsafe { vaddvq_f32(acc) };
        for (av, bv) in a[offset..].iter().zip(b[offset..].iter()) {
            result += av.to_f32() * bv;
        }
        result
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(av, bv)| av.to_f32() * bv).sum()
    }
}

// ── accumulate ─────────────────────────────────────────────────────

/// Accumulate `f16` values into an `f32` buffer: `dst[i] += src[i]`.
///
/// Converts each `f16` to `f32` before adding. Uses NEON FADD
/// on aarch64; scalar path elsewhere.
///
/// # Panics
///
/// Panics if `dst.len() < src.len()`.
pub fn accumulate_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    assert!(dst.len() >= src.len(), "accumulate: dst len {} < src len {}", dst.len(), src.len(),);
    let len = src.len();

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut offset = 0;
        for chunk in src.chunks_exact(4) {
            let buf = [chunk[0].to_f32(), chunk[1].to_f32(), chunk[2].to_f32(), chunk[3].to_f32()];
            unsafe {
                let s_fv = vld1q_f32(buf.as_ptr());
                let dp = dst.as_mut_ptr().add(offset);
                let d_fv = vld1q_f32(dp);
                let sum = vaddq_f32(d_fv, s_fv);
                vst1q_f32(dp, sum);
            }
            offset += 4;
        }
        for (d, s) in dst[offset..len].iter_mut().zip(src[offset..].iter()) {
            *d += s.to_f32();
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = len;
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += s.to_f32();
        }
    }
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_F16: f32 = 1e-3;
    const EPS_BF16: f32 = 1e-2;
    const EPS_DOT: f32 = 5e-2;

    fn f16s(vals: &[f32]) -> Vec<f16> {
        vals.iter().map(|&v| f16::from_f32(v)).collect()
    }

    fn assert_approx(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < eps || (x.is_nan() && y.is_nan()),
                "index {i}: {x} vs {y} (eps={eps})"
            );
        }
    }

    // ── f16 → f32 (15 tests) ──────────────────────────────────

    #[test]
    fn test_f16_to_f32_empty() {
        assert!(f16_to_f32_neon(&[]).is_empty());
    }

    #[test]
    fn test_f16_to_f32_single_zero() {
        let r = f16_to_f32_neon(&f16s(&[0.0]));
        assert_approx(&r, &[0.0], EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_single_one() {
        let r = f16_to_f32_neon(&f16s(&[1.0]));
        assert_approx(&r, &[1.0], EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_single_negative() {
        let r = f16_to_f32_neon(&f16s(&[-1.0]));
        assert_approx(&r, &[-1.0], EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_four_aligned() {
        let inp = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let r = f16_to_f32_neon(&inp);
        assert_approx(&r, &[1.0, 2.0, 3.0, 4.0], EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_eight_elements() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let r = f16_to_f32_neon(&f16s(&v));
        assert_approx(&r, &v, EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_non_aligned_5() {
        let v: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let r = f16_to_f32_neon(&f16s(&v));
        assert_approx(&r, &v, EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_non_aligned_7() {
        let v: Vec<f32> = (1..=7).map(|i| i as f32).collect();
        let r = f16_to_f32_neon(&f16s(&v));
        assert_approx(&r, &v, EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_large_16() {
        let v: Vec<f32> = (1..=16).map(|i| i as f32 * 0.5).collect();
        let r = f16_to_f32_neon(&f16s(&v));
        assert_approx(&r, &v, EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_large_33() {
        let v: Vec<f32> = (1..=33).map(|i| i as f32 * 0.1).collect();
        let r = f16_to_f32_neon(&f16s(&v));
        for (a, b) in r.iter().zip(v.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_f16_to_f32_all_zeros() {
        let inp = vec![f16::ZERO; 12];
        let r = f16_to_f32_neon(&inp);
        assert!(r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_f16_to_f32_mixed_signs() {
        let inp = f16s(&[-2.0, 1.0, -0.5, 3.0, -4.0]);
        let r = f16_to_f32_neon(&inp);
        assert_approx(&r, &[-2.0, 1.0, -0.5, 3.0, -4.0], EPS_F16);
    }

    #[test]
    fn test_f16_to_f32_max_value() {
        let inp = [f16::MAX];
        let r = f16_to_f32_neon(&inp);
        assert!((r[0] - f16::MAX.to_f32()).abs() < 1.0);
    }

    #[test]
    fn test_f16_to_f32_min_positive() {
        let inp = [f16::MIN_POSITIVE];
        let r = f16_to_f32_neon(&inp);
        assert!(r[0] > 0.0);
    }

    #[test]
    fn test_f16_to_f32_preserves_sign() {
        let inp = f16s(&[-5.0, 5.0, -0.0, 0.0]);
        let r = f16_to_f32_neon(&inp);
        assert!(r[0] < 0.0);
        assert!(r[1] > 0.0);
    }

    // ── f32 → f16 (13 tests) ──────────────────────────────────

    #[test]
    fn test_f32_to_f16_empty() {
        assert!(f32_to_f16_neon(&[]).is_empty());
    }

    #[test]
    fn test_f32_to_f16_single_zero() {
        let r = f32_to_f16_neon(&[0.0]);
        assert_eq!(r[0].to_f32(), 0.0);
    }

    #[test]
    fn test_f32_to_f16_single_one() {
        let r = f32_to_f16_neon(&[1.0]);
        assert!((r[0].to_f32() - 1.0).abs() < EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_single_negative() {
        let r = f32_to_f16_neon(&[-1.0]);
        assert!((r[0].to_f32() + 1.0).abs() < EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_four_aligned() {
        let r = f32_to_f16_neon(&[1.0, 2.0, 3.0, 4.0]);
        let f: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
        assert_approx(&f, &[1.0, 2.0, 3.0, 4.0], EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_eight_elements() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let r = f32_to_f16_neon(&v);
        let f: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
        assert_approx(&f, &v, EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_non_aligned_5() {
        let v: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let r = f32_to_f16_neon(&v);
        assert_eq!(r.len(), 5);
        let f: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
        assert_approx(&f, &v, EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_large_16() {
        let v: Vec<f32> = (1..=16).map(|i| i as f32 * 0.5).collect();
        let r = f32_to_f16_neon(&v);
        let f: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
        assert_approx(&f, &v, EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_all_zeros() {
        let r = f32_to_f16_neon(&[0.0; 12]);
        assert!(r.iter().all(|v| v.to_f32() == 0.0));
    }

    #[test]
    fn test_f32_to_f16_mixed_values() {
        let v = [-2.0f32, 0.5, 100.0, -0.25, 1.0];
        let r = f32_to_f16_neon(&v);
        let f: Vec<f32> = r.iter().map(|v| v.to_f32()).collect();
        assert_approx(&f, &v, 0.1);
    }

    #[test]
    fn test_f32_to_f16_large_clamps_to_inf() {
        let r = f32_to_f16_neon(&[1e38]);
        assert!(r[0].to_f32().is_infinite());
    }

    #[test]
    fn test_f32_to_f16_roundtrip() {
        let v = [1.0f32, 2.0, 0.5, -1.0, 0.0, -0.5, 4.0, 8.0];
        let h = f32_to_f16_neon(&v);
        let back = f16_to_f32_neon(&h);
        assert_approx(&back, &v, EPS_F16);
    }

    #[test]
    fn test_f32_to_f16_preserves_sign() {
        let r = f32_to_f16_neon(&[-3.0, 3.0]);
        assert!(r[0].to_f32() < 0.0);
        assert!(r[1].to_f32() > 0.0);
    }

    // ── bf16 → f32 (12 tests) ─────────────────────────────────

    #[test]
    fn test_bf16_to_f32_empty() {
        assert!(bf16_to_f32_neon(&[]).is_empty());
    }

    #[test]
    fn test_bf16_to_f32_single_zero() {
        let r = bf16_to_f32_neon(&[0x0000]);
        assert_eq!(r[0], 0.0);
    }

    #[test]
    fn test_bf16_to_f32_single_one() {
        let r = bf16_to_f32_neon(&[0x3F80]);
        assert_eq!(r[0], 1.0);
    }

    #[test]
    fn test_bf16_to_f32_four_aligned() {
        let inp = [0x3F80, 0x4000, 0x3F00, 0xBF80];
        let r = bf16_to_f32_neon(&inp);
        assert_approx(&r, &[1.0, 2.0, 0.5, -1.0], EPS_BF16);
    }

    #[test]
    fn test_bf16_to_f32_eight_elements() {
        let inp = [0x3F80, 0x4000, 0x4040, 0x4080, 0x3F00, 0xBF80, 0xC000, 0x0000];
        let expected = [1.0, 2.0, 3.0, 4.0, 0.5, -1.0, -2.0, 0.0];
        let r = bf16_to_f32_neon(&inp);
        assert_approx(&r, &expected, EPS_BF16);
    }

    #[test]
    fn test_bf16_to_f32_non_aligned_5() {
        let inp = [0x3F80, 0x4000, 0x4040, 0x4080, 0x3F00];
        let r = bf16_to_f32_neon(&inp);
        assert_eq!(r.len(), 5);
        assert_approx(&r, &[1.0, 2.0, 3.0, 4.0, 0.5], EPS_BF16);
    }

    #[test]
    fn test_bf16_to_f32_large_16() {
        let inp = vec![0x3F80u16; 16];
        let r = bf16_to_f32_neon(&inp);
        assert!(r.iter().all(|&v| (v - 1.0).abs() < EPS_BF16));
    }

    #[test]
    fn test_bf16_to_f32_all_zeros() {
        let inp = vec![0u16; 12];
        let r = bf16_to_f32_neon(&inp);
        assert!(r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bf16_to_f32_negative_values() {
        let inp = [0xBF80, 0xC000, 0xBF00];
        let r = bf16_to_f32_neon(&inp);
        assert_approx(&r, &[-1.0, -2.0, -0.5], EPS_BF16);
    }

    #[test]
    fn test_bf16_to_f32_mixed_values() {
        let inp = [0x4120, 0xC120, 0x0000];
        let r = bf16_to_f32_neon(&inp);
        assert!(r[0] > 9.0 && r[0] < 11.0);
        assert!(r[1] < -9.0 && r[1] > -11.0);
        assert_eq!(r[2], 0.0);
    }

    #[test]
    fn test_bf16_to_f32_half_value() {
        let r = bf16_to_f32_neon(&[0x3F00]);
        assert_eq!(r[0], 0.5);
    }

    #[test]
    fn test_bf16_to_f32_large_33() {
        let inp = vec![0x4000u16; 33];
        let r = bf16_to_f32_neon(&inp);
        assert_eq!(r.len(), 33);
        assert!(r.iter().all(|&v| (v - 2.0).abs() < EPS_BF16));
    }

    // ── f32 → bf16 (12 tests) ─────────────────────────────────

    #[test]
    fn test_f32_to_bf16_empty() {
        assert!(f32_to_bf16_neon(&[]).is_empty());
    }

    #[test]
    fn test_f32_to_bf16_single_zero() {
        let r = f32_to_bf16_neon(&[0.0]);
        assert_eq!(r[0], 0x0000);
    }

    #[test]
    fn test_f32_to_bf16_single_one() {
        let r = f32_to_bf16_neon(&[1.0]);
        assert_eq!(r[0], 0x3F80);
    }

    #[test]
    fn test_f32_to_bf16_four_aligned() {
        let r = f32_to_bf16_neon(&[1.0, 2.0, 0.5, -1.0]);
        assert_eq!(r, [0x3F80, 0x4000, 0x3F00, 0xBF80]);
    }

    #[test]
    fn test_f32_to_bf16_eight_elements() {
        let v = [1.0f32, 2.0, 3.0, 4.0, 0.5, -1.0, -2.0, 0.0];
        let r = f32_to_bf16_neon(&v);
        let expected = [0x3F80, 0x4000, 0x4040, 0x4080, 0x3F00, 0xBF80, 0xC000, 0x0000];
        assert_eq!(r, expected);
    }

    #[test]
    fn test_f32_to_bf16_non_aligned_5() {
        let r = f32_to_bf16_neon(&[1.0, 2.0, 3.0, 4.0, 0.5]);
        assert_eq!(r.len(), 5);
        assert_eq!(r, [0x3F80, 0x4000, 0x4040, 0x4080, 0x3F00]);
    }

    #[test]
    fn test_f32_to_bf16_large_16() {
        let v = vec![1.0f32; 16];
        let r = f32_to_bf16_neon(&v);
        assert!(r.iter().all(|&b| b == 0x3F80));
    }

    #[test]
    fn test_f32_to_bf16_all_zeros() {
        let r = f32_to_bf16_neon(&[0.0; 12]);
        assert!(r.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_f32_to_bf16_negative_values() {
        let r = f32_to_bf16_neon(&[-1.0, -2.0, -0.5]);
        assert_eq!(r, [0xBF80, 0xC000, 0xBF00]);
    }

    #[test]
    fn test_f32_to_bf16_mixed_values() {
        let r = f32_to_bf16_neon(&[1.0, -1.0, 0.0]);
        assert_eq!(r[0], 0x3F80);
        assert_eq!(r[1], 0xBF80);
        assert_eq!(r[2], 0x0000);
    }

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let v = [1.0f32, 2.0, 0.5, -1.0, 0.0, -0.5, 4.0, 8.0];
        let bf = f32_to_bf16_neon(&v);
        let back = bf16_to_f32_neon(&bf);
        assert_approx(&back, &v, EPS_BF16);
    }

    #[test]
    fn test_f32_to_bf16_large_33() {
        let v = vec![2.0f32; 33];
        let r = f32_to_bf16_neon(&v);
        assert_eq!(r.len(), 33);
        assert!(r.iter().all(|&b| b == 0x4000));
    }

    // ── mixed dot product (16 tests) ──────────────────────────

    #[test]
    fn test_mixed_dot_empty() {
        assert_eq!(mixed_dot_f16_f32(&[], &[]), 0.0);
    }

    #[test]
    fn test_mixed_dot_single() {
        let a = f16s(&[2.0]);
        let r = mixed_dot_f16_f32(&a, &[3.0]);
        assert!((r - 6.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_four_aligned() {
        let a = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let b = [1.0f32, 1.0, 1.0, 1.0];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 10.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_eight_aligned() {
        let a = f16s(&[1.0; 8]);
        let b = [2.0f32; 8];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 16.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_non_aligned_5() {
        let a = f16s(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = [1.0f32; 5];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 15.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_non_aligned_7() {
        let a = f16s(&[1.0; 7]);
        let b = [3.0f32; 7];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 21.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_large_16() {
        let a = f16s(&[0.5; 16]);
        let b = [2.0f32; 16];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 16.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_large_33() {
        let a = f16s(&[1.0; 33]);
        let b = [1.0f32; 33];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 33.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_all_zeros() {
        let a = f16s(&[0.0; 8]);
        let b = [0.0f32; 8];
        assert_eq!(mixed_dot_f16_f32(&a, &b), 0.0);
    }

    #[test]
    fn test_mixed_dot_one_side_zero() {
        let a = f16s(&[0.0; 4]);
        let b = [5.0f32; 4];
        assert!(mixed_dot_f16_f32(&a, &b).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_negative() {
        let a = f16s(&[-1.0, -2.0, -3.0, -4.0]);
        let b = [1.0f32, 1.0, 1.0, 1.0];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r + 10.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_mixed_signs() {
        let a = f16s(&[1.0, -1.0, 1.0, -1.0]);
        let b = [1.0f32, 1.0, 1.0, 1.0];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!(r.abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_unit_like() {
        let a = f16s(&[1.0, 0.0, 0.0]);
        let b = [5.0f32, 99.0, 99.0];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 5.0).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_large_values() {
        let a = f16s(&[100.0, 200.0]);
        let b = [0.01f32, 0.01];
        let r = mixed_dot_f16_f32(&a, &b);
        assert!((r - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_mixed_dot_commutativity() {
        let av = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bv = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let r1 = mixed_dot_f16_f32(&f16s(&av), &bv);
        let r2 = mixed_dot_f16_f32(&f16s(&bv), &av);
        assert!((r1 - r2).abs() < EPS_DOT);
    }

    #[test]
    fn test_mixed_dot_distributivity() {
        let a = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let b = f16s(&[4.0, 3.0, 2.0, 1.0]);
        let c = [1.0f32; 4];
        let ab: Vec<f16> =
            a.iter().zip(b.iter()).map(|(x, y)| f16::from_f32(x.to_f32() + y.to_f32())).collect();
        let lhs = mixed_dot_f16_f32(&ab, &c);
        let rhs = mixed_dot_f16_f32(&a, &c) + mixed_dot_f16_f32(&b, &c);
        assert!((lhs - rhs).abs() < 0.5);
    }

    // ── accumulate (15 tests) ─────────────────────────────────

    #[test]
    fn test_accumulate_empty() {
        let mut dst = [1.0f32; 4];
        accumulate_f16_to_f32(&[], &mut dst);
        assert_eq!(dst, [1.0; 4]);
    }

    #[test]
    fn test_accumulate_single() {
        let src = f16s(&[2.0]);
        let mut dst = vec![1.0f32];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!((dst[0] - 3.0).abs() < EPS_F16);
    }

    #[test]
    fn test_accumulate_four_aligned() {
        let src = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let mut dst = vec![0.0f32; 4];
        accumulate_f16_to_f32(&src, &mut dst);
        assert_approx(&dst, &[1.0, 2.0, 3.0, 4.0], EPS_F16);
    }

    #[test]
    fn test_accumulate_eight_elements() {
        let src = f16s(&[1.0; 8]);
        let mut dst = vec![1.0f32; 8];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!(dst.iter().all(|&v| (v - 2.0).abs() < EPS_F16));
    }

    #[test]
    fn test_accumulate_non_aligned_5() {
        let src = f16s(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut dst = vec![0.0f32; 5];
        accumulate_f16_to_f32(&src, &mut dst);
        let expected = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_approx(&dst, &expected, EPS_F16);
    }

    #[test]
    fn test_accumulate_large_16() {
        let src = f16s(&[0.5; 16]);
        let mut dst = vec![0.5f32; 16];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!(dst.iter().all(|&v| (v - 1.0).abs() < EPS_F16));
    }

    #[test]
    fn test_accumulate_large_33() {
        let src = f16s(&[1.0; 33]);
        let mut dst = vec![0.0f32; 33];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!(dst.iter().all(|&v| (v - 1.0).abs() < EPS_F16));
    }

    #[test]
    fn test_accumulate_all_zeros_src() {
        let src = vec![f16::ZERO; 8];
        let mut dst = vec![5.0f32; 8];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!(dst.iter().all(|&v| (v - 5.0).abs() < EPS_F16));
    }

    #[test]
    fn test_accumulate_into_existing() {
        let src = f16s(&[10.0, 20.0, 30.0, 40.0]);
        let mut dst = vec![1.0, 2.0, 3.0, 4.0f32];
        accumulate_f16_to_f32(&src, &mut dst);
        assert_approx(&dst, &[11.0, 22.0, 33.0, 44.0], 0.1);
    }

    #[test]
    fn test_accumulate_negative() {
        let src = f16s(&[-1.0, -2.0, -3.0, -4.0]);
        let mut dst = vec![10.0f32; 4];
        accumulate_f16_to_f32(&src, &mut dst);
        assert_approx(&dst, &[9.0, 8.0, 7.0, 6.0], EPS_F16);
    }

    #[test]
    fn test_accumulate_mixed() {
        let src = f16s(&[-1.0, 1.0, -1.0, 1.0, -1.0]);
        let mut dst = vec![0.0f32; 5];
        accumulate_f16_to_f32(&src, &mut dst);
        let expected = [-1.0f32, 1.0, -1.0, 1.0, -1.0];
        assert_approx(&dst, &expected, EPS_F16);
    }

    #[test]
    fn test_accumulate_twice() {
        let src = f16s(&[1.0; 4]);
        let mut dst = vec![0.0f32; 4];
        accumulate_f16_to_f32(&src, &mut dst);
        accumulate_f16_to_f32(&src, &mut dst);
        assert!(dst.iter().all(|&v| (v - 2.0).abs() < EPS_F16));
    }

    #[test]
    fn test_accumulate_longer_dst() {
        let src = f16s(&[1.0, 2.0]);
        let mut dst = vec![0.0f32; 5];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!((dst[0] - 1.0).abs() < EPS_F16);
        assert!((dst[1] - 2.0).abs() < EPS_F16);
        assert_eq!(dst[2], 0.0);
        assert_eq!(dst[3], 0.0);
        assert_eq!(dst[4], 0.0);
    }

    #[test]
    fn test_accumulate_preserves_tail() {
        let src = f16s(&[1.0]);
        let mut dst = vec![0.0, 99.0, 99.0f32];
        accumulate_f16_to_f32(&src, &mut dst);
        assert!((dst[0] - 1.0).abs() < EPS_F16);
        assert_eq!(dst[1], 99.0);
        assert_eq!(dst[2], 99.0);
    }

    #[test]
    fn test_accumulate_additive() {
        let a = f16s(&[1.0, 2.0, 3.0, 4.0]);
        let b = f16s(&[4.0, 3.0, 2.0, 1.0]);
        let mut d = vec![0.0f32; 4];
        accumulate_f16_to_f32(&a, &mut d);
        accumulate_f16_to_f32(&b, &mut d);
        assert!(d.iter().all(|&v| (v - 5.0).abs() < EPS_F16));
    }

    // ── cross-function (5 tests) ──────────────────────────────

    #[test]
    fn test_f16_f32_roundtrip_many() {
        let v: Vec<f32> = (0..20).map(|i| (i as f32) * 0.25 - 2.5).collect();
        let h = f32_to_f16_neon(&v);
        let back = f16_to_f32_neon(&h);
        assert_approx(&back, &v, 0.01);
    }

    #[test]
    fn test_bf16_f32_roundtrip_many() {
        let v = [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 4.0];
        let bf = f32_to_bf16_neon(&v);
        let back = bf16_to_f32_neon(&bf);
        assert_approx(&back, &v, EPS_BF16);
    }

    #[test]
    fn test_dot_matches_manual() {
        let av = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bv = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let manual: f32 = av.iter().zip(bv.iter()).map(|(a, b)| a * b).sum();
        let r = mixed_dot_f16_f32(&f16s(&av), &bv);
        assert!((r - manual).abs() < EPS_DOT);
    }

    #[test]
    fn test_accumulate_matches_convert_add() {
        let src = f16s(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let init = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut dst = init.to_vec();
        accumulate_f16_to_f32(&src, &mut dst);
        let converted = f16_to_f32_neon(&src);
        let expected: Vec<f32> = init.iter().zip(converted.iter()).map(|(a, b)| a + b).collect();
        assert_approx(&dst, &expected, EPS_F16);
    }

    #[test]
    fn test_conversion_identity_zero() {
        let z16 = f32_to_f16_neon(&[0.0]);
        assert_eq!(f16_to_f32_neon(&z16)[0], 0.0);
        let zbf = f32_to_bf16_neon(&[0.0]);
        assert_eq!(bf16_to_f32_neon(&zbf)[0], 0.0);
    }
}
