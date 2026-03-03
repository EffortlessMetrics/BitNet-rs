//! Data type conversion utilities.
//!
//! Efficient conversion between floating-point formats (FP32, FP16, BF16)
//! commonly used in neural network weight loading and inference.

/// Convert a BF16 (bfloat16) value to f32.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert an f32 value to BF16 (bfloat16) with round-to-nearest-even.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Handle NaN: preserve NaN payload
    if val.is_nan() {
        return ((bits >> 16) | 0x0040) as u16; // quiet NaN
    }
    // Round to nearest even
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convert an IEEE 754 FP16 (half) value to f32.
#[inline]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31); // +/- zero
        }
        // Denormalized: convert to normalized f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        if mantissa == 0 {
            return f32::from_bits((sign << 31) | 0x7F800000); // +/- inf
        }
        return f32::from_bits((sign << 31) | 0x7FC00000 | (mantissa << 13)); // NaN
    }
    // Normalized
    let f32_exp = (exponent + 112) & 0xFF; // -15 + 127 = 112 bias adjustment
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Convert an f32 value to FP16.
#[inline]
pub fn f32_to_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 255 {
        // Inf or NaN
        if mantissa == 0 {
            return (sign << 15) | 0x7C00; // Inf
        }
        return (sign << 15) | 0x7E00; // NaN
    }

    let new_exp = exponent - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00; // overflow to inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15; // underflow to zero
        }
        // Denormalized
        let m = (mantissa | 0x800000) >> (1 - new_exp + 13);
        return (sign << 15) | (m as u16);
    }
    // Normalized
    let round = (mantissa >> 12) & 1;
    let m = ((mantissa >> 13) + round).min(0x3FF);
    (sign << 15) | ((new_exp as u16) << 10) | (m as u16)
}

/// Batch convert BF16 slice to f32 vec.
pub fn bf16_slice_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().copied().map(bf16_to_f32).collect()
}

/// Batch convert f32 slice to BF16 vec.
pub fn f32_slice_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter().copied().map(f32_to_bf16).collect()
}

/// Batch convert FP16 slice to f32 vec.
pub fn fp16_slice_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().copied().map(fp16_to_f32).collect()
}

/// Batch convert f32 slice to FP16 vec.
pub fn f32_slice_to_fp16(data: &[f32]) -> Vec<u16> {
    data.iter().copied().map(f32_to_fp16).collect()
}

/// Supported data types for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 3.15, 100.0, -0.5];
        for &v in &vals {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            assert!((back - v).abs() < 0.02, "bf16 roundtrip failed for {v}: got {back}");
        }
    }

    #[test]
    fn test_fp16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, -0.5];
        for &v in &vals {
            let fp = f32_to_fp16(v);
            let back = fp16_to_f32(fp);
            assert!((back - v).abs() < 0.001, "fp16 roundtrip failed for {v}: got {back}");
        }
    }

    #[test]
    fn test_bf16_zero() {
        assert_eq!(bf16_to_f32(0), 0.0);
        assert_eq!(f32_to_bf16(0.0), 0);
    }

    #[test]
    fn test_bf16_negative_zero() {
        let nz = f32_to_bf16(-0.0);
        let back = bf16_to_f32(nz);
        assert!(back == 0.0 && back.is_sign_negative());
    }

    #[test]
    fn test_fp16_inf() {
        let inf_bits = f32_to_fp16(f32::INFINITY);
        assert_eq!(inf_bits, 0x7C00);
        let back = fp16_to_f32(0x7C00);
        assert!(back.is_infinite() && back.is_sign_positive());
    }

    #[test]
    fn test_fp16_neg_inf() {
        let neg_inf_bits = f32_to_fp16(f32::NEG_INFINITY);
        assert_eq!(neg_inf_bits, 0xFC00);
    }

    #[test]
    fn test_fp16_nan() {
        let nan_bits = f32_to_fp16(f32::NAN);
        let back = fp16_to_f32(nan_bits);
        assert!(back.is_nan());
    }

    #[test]
    fn test_bf16_nan() {
        let nan_bits = f32_to_bf16(f32::NAN);
        let back = bf16_to_f32(nan_bits);
        assert!(back.is_nan());
    }

    #[test]
    fn test_batch_bf16_to_f32() {
        let bf16s: Vec<u16> = vec![0x3F80, 0x4000, 0xBF80]; // 1.0, 2.0, -1.0
        let f32s = bf16_slice_to_f32(&bf16s);
        assert!((f32s[0] - 1.0).abs() < 0.01);
        assert!((f32s[1] - 2.0).abs() < 0.01);
        assert!((f32s[2] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_batch_f32_to_bf16() {
        let f32s = vec![1.0f32, 2.0, -1.0];
        let bf16s = f32_slice_to_bf16(&f32s);
        assert_eq!(bf16s.len(), 3);
        assert_eq!(bf16s[0], 0x3F80);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(DType::F32.name(), "float32");
        assert_eq!(DType::BF16.name(), "bfloat16");
    }
}
