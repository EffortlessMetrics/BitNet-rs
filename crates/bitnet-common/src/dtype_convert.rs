//! Data type conversion utilities for model weights.
//!
//! Handles BF16↔F32, F16↔F32, and related conversions needed
//! when loading models with different weight precisions.

/// Convert BF16 (as u16) to F32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert F32 to BF16 (round to nearest even).
#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    // Round to nearest even
    let round_bit = 1u32 << 15;
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(round_bit - 1 + lsb);
    (rounded >> 16) as u16
}

/// Convert F16 (IEEE 754 half) to F32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: convert to normal f32
            let mut m = mant;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let f32_exp = (127 - 15 - e) as u32;
            let f32_mant = (m & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
        }
    } else if exp == 31 {
        // Inf or NaN
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normal
        let f32_exp = (exp + 127 - 15) << 23;
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_mant)
    }
}

/// Convert F32 to F16.
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        let f16_mant = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7C00 | f16_mant;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow -> Inf
        return (sign << 15) | 0x7C00;
    }
    if unbiased < -24 {
        // Underflow -> zero
        return sign << 15;
    }
    if unbiased < -14 {
        // Subnormal
        let shift = -14 - unbiased;
        let f16_mant = ((0x800000 | mant) >> (shift + 13)) as u16;
        return (sign << 15) | f16_mant;
    }

    let f16_exp = ((unbiased + 15) as u16) << 10;
    let f16_mant = (mant >> 13) as u16;
    (sign << 15) | f16_exp | f16_mant
}

/// Batch convert BF16 slice to F32.
pub fn bf16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = bf16_to_f32(*s);
    }
}

/// Batch convert F32 slice to BF16.
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f32_to_bf16(*s);
    }
}

/// Batch convert F16 slice to F32.
pub fn f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = f16_to_f32(*s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_round_trip() {
        let val = 3.15f32;
        let bf16 = f32_to_bf16(val);
        let back = bf16_to_f32(bf16);
        assert!((val - back).abs() < 0.02);
    }

    #[test]
    fn test_bf16_zero() {
        assert_eq!(bf16_to_f32(0), 0.0);
        assert_eq!(f32_to_bf16(0.0), 0);
    }

    #[test]
    fn test_bf16_one() {
        let bf16 = f32_to_bf16(1.0);
        assert!((bf16_to_f32(bf16) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bf16_negative() {
        let bf16 = f32_to_bf16(-2.5);
        assert!((bf16_to_f32(bf16) + 2.5).abs() < 0.02);
    }

    #[test]
    fn test_f16_round_trip() {
        let val = 1.5f32;
        let f16 = f32_to_f16(val);
        let back = f16_to_f32(f16);
        assert!((val - back).abs() < 0.001);
    }

    #[test]
    fn test_f16_zero() {
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f32_to_f16(0.0), 0);
    }

    #[test]
    fn test_f16_inf() {
        let f16 = f32_to_f16(f32::INFINITY);
        assert!(f16_to_f32(f16).is_infinite());
    }

    #[test]
    fn test_f16_nan() {
        let f16 = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(f16).is_nan());
    }

    #[test]
    fn test_f16_overflow() {
        let f16 = f32_to_f16(100000.0);
        assert!(f16_to_f32(f16).is_infinite());
    }

    #[test]
    fn test_f16_underflow() {
        let f16 = f32_to_f16(1e-40);
        assert_eq!(f16_to_f32(f16), 0.0);
    }

    #[test]
    fn test_f16_subnormal() {
        let f16 = f32_to_f16(5.96e-8); // smallest f16 subnormal
        let back = f16_to_f32(f16);
        assert!(back >= 0.0 && back < 1e-6);
    }

    #[test]
    fn test_bf16_slice() {
        let src = vec![1.0f32, 2.0, 3.0];
        let mut bf16 = vec![0u16; 3];
        let mut dst = vec![0.0f32; 3];
        f32_to_bf16_slice(&src, &mut bf16);
        bf16_to_f32_slice(&bf16, &mut dst);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 0.02);
        }
    }

    #[test]
    fn test_f16_slice() {
        let src = vec![1.0f32, -1.0, 0.5];
        let mut f16 = vec![0u16; 3];
        let mut dst = vec![0.0f32; 3];
        for (i, &v) in src.iter().enumerate() {
            f16[i] = f32_to_f16(v);
        }
        f16_to_f32_slice(&f16, &mut dst);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_bf16_neg_zero() {
        let nz = f32_to_bf16(-0.0);
        let back = bf16_to_f32(nz);
        assert_eq!(back, 0.0); // -0.0 == 0.0 in f32
    }

    #[test]
    fn test_f16_negative() {
        let f16 = f32_to_f16(-3.0);
        assert!((f16_to_f32(f16) + 3.0).abs() < 0.01);
    }
}
