//! Low-level 2-bit packing and unpacking.
//!
//! The public [`pack_i2s`] / [`unpack_i2s`] functions operate on ternary `i8`
//! slices (`{-1, 0, +1}`) and dispatch to SIMD where available.

use crate::error::I2SError;
use crate::simd;

/// Code mapping: ternary value to 2-bit code.
///
/// - -1 maps to 0
/// -  0 maps to 1
/// - +1 maps to 2
#[allow(clippy::cast_sign_loss)]
#[inline]
const fn ternary_to_code(v: i8) -> u8 {
    let clamped = if v < -1 {
        -1
    } else if v > 1 {
        1
    } else {
        v
    };
    (clamped + 1) as u8
}

/// Code mapping: 2-bit code to ternary value.
#[allow(clippy::cast_possible_wrap)]
#[inline]
const fn code_to_ternary(c: u8) -> i8 {
    // Codes: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> +1 (saturate)
    let capped = if c > 2 { 2 } else { c };
    (capped as i8) - 1
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

fn pack_scalar(values: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(4));
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &v) in chunk.iter().enumerate() {
            byte |= ternary_to_code(v) << (i * 2);
        }
        out.push(byte);
    }
    out
}

fn unpack_scalar(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for &byte in packed {
        for shift in (0..8).step_by(2) {
            if out.len() >= count {
                break;
            }
            out.push(code_to_ternary((byte >> shift) & 0x3));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Pack ternary `i8` values (`{-1, 0, +1}`) into 2-bit packed bytes.
///
/// Returns `ceil(values.len() / 4)` bytes.
///
/// Values outside `[-1, 1]` are clamped.
///
/// # Errors
///
/// Returns [`I2SError::EmptyInput`] if `values` is empty.
pub fn pack_i2s(values: &[i8]) -> Result<Vec<u8>, I2SError> {
    if values.is_empty() {
        return Err(I2SError::EmptyInput);
    }

    // SIMD fast paths
    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() && values.len() >= 32 {
        return Ok(pack_avx2_chunked(values));
    }

    #[cfg(target_arch = "aarch64")]
    if simd::has_neon() && values.len() >= 16 {
        return Ok(pack_neon_chunked(values));
    }

    Ok(pack_scalar(values))
}

/// Unpack 2-bit packed bytes into ternary `i8` values.
///
/// # Errors
///
/// Returns [`I2SError::EmptyInput`] if `packed` is empty.
///
/// Returns [`I2SError::PackedLengthMismatch`] if the packed length is too
/// small for the requested `count`.
pub fn unpack_i2s(packed: &[u8], count: usize) -> Result<Vec<i8>, I2SError> {
    if packed.is_empty() || count == 0 {
        return Err(I2SError::EmptyInput);
    }
    let needed = count.div_ceil(4);
    if packed.len() < needed {
        return Err(I2SError::PackedLengthMismatch { actual: packed.len(), expected: needed });
    }

    #[cfg(target_arch = "x86_64")]
    if simd::has_avx2() && count >= 32 {
        return Ok(unpack_avx2_chunked(packed, count));
    }

    #[cfg(target_arch = "aarch64")]
    if simd::has_neon() && count >= 16 {
        return Ok(unpack_neon_chunked(packed, count));
    }

    Ok(unpack_scalar(packed, count))
}

// ---------------------------------------------------------------------------
// AVX2 chunked dispatch
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn pack_avx2_chunked(values: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(4));
    let chunks = values.chunks_exact(32);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: [i8; 32] = chunk.try_into().unwrap();
        let packed = unsafe { simd::avx2::pack_32_avx2(&arr) };
        out.extend_from_slice(&packed);
    }
    if !remainder.is_empty() {
        out.extend_from_slice(&pack_scalar(remainder));
    }
    out
}

#[cfg(target_arch = "x86_64")]
fn unpack_avx2_chunked(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    let full_chunks = count / 32;
    let remainder = count % 32;

    for i in 0..full_chunks {
        let arr: [u8; 8] = packed[i * 8..(i + 1) * 8].try_into().unwrap();
        let unpacked = unsafe { simd::avx2::unpack_32_avx2(&arr) };
        out.extend_from_slice(&unpacked);
    }
    if remainder > 0 {
        let start = full_chunks * 8;
        out.extend_from_slice(&unpack_scalar(&packed[start..], remainder));
    }
    out
}

// ---------------------------------------------------------------------------
// NEON chunked dispatch
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn pack_neon_chunked(values: &[i8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(4));
    let chunks = values.chunks_exact(16);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: [i8; 16] = chunk.try_into().unwrap();
        let packed = unsafe { simd::neon::pack_16_neon(&arr) };
        out.extend_from_slice(&packed);
    }
    if !remainder.is_empty() {
        out.extend_from_slice(&pack_scalar(remainder));
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn unpack_neon_chunked(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    let full_chunks = count / 16;
    let remainder = count % 16;

    for i in 0..full_chunks {
        let arr: [u8; 4] = packed[i * 4..(i + 1) * 4].try_into().unwrap();
        let unpacked = unsafe { simd::neon::unpack_16_neon(&arr) };
        out.extend_from_slice(&unpacked);
    }
    if remainder > 0 {
        let start = full_chunks * 4;
        out.extend_from_slice(&unpack_scalar(&packed[start..], remainder));
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Code mapping --------------------------------------------------------

    #[test]
    fn code_mapping_ternary_to_code() {
        assert_eq!(ternary_to_code(-1), 0);
        assert_eq!(ternary_to_code(0), 1);
        assert_eq!(ternary_to_code(1), 2);
    }

    #[test]
    fn code_mapping_clamping() {
        assert_eq!(ternary_to_code(-100), 0); // clamp to -1 -> code 0
        assert_eq!(ternary_to_code(100), 2); // clamp to +1 -> code 2
    }

    #[test]
    fn code_mapping_code_to_ternary() {
        assert_eq!(code_to_ternary(0), -1);
        assert_eq!(code_to_ternary(1), 0);
        assert_eq!(code_to_ternary(2), 1);
        assert_eq!(code_to_ternary(3), 1); // saturate
    }

    // -- Scalar pack/unpack --------------------------------------------------

    #[test]
    fn scalar_pack_4_values() {
        let v = [1i8, 0, -1, 1];
        let p = pack_scalar(&v);
        assert_eq!(p.len(), 1);
        // codes: 2, 1, 0, 2 -> byte = 2 | (1<<2) | (0<<4) | (2<<6) = 0x86
        assert_eq!(p[0], 0x86);
    }

    #[test]
    fn scalar_roundtrip() {
        let values: Vec<i8> = vec![-1, 0, 1, -1, 0, 1, -1];
        let packed = pack_scalar(&values);
        let unpacked = unpack_scalar(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn scalar_roundtrip_all_neg1() {
        let values = vec![-1i8; 16];
        let packed = pack_scalar(&values);
        let unpacked = unpack_scalar(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn scalar_roundtrip_all_zero() {
        let values = vec![0i8; 16];
        let packed = pack_scalar(&values);
        let unpacked = unpack_scalar(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    #[test]
    fn scalar_roundtrip_all_one() {
        let values = vec![1i8; 16];
        let packed = pack_scalar(&values);
        let unpacked = unpack_scalar(&packed, values.len());
        assert_eq!(unpacked, values);
    }

    // -- Public API ----------------------------------------------------------

    #[test]
    fn pack_unpack_basic() {
        let values: Vec<i8> = vec![-1, 0, 1, 0, -1, 1];
        let packed = pack_i2s(&values).unwrap();
        let unpacked = unpack_i2s(&packed, values.len()).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_empty_errors() {
        assert!(pack_i2s(&[]).is_err());
    }

    #[test]
    fn unpack_empty_errors() {
        assert!(unpack_i2s(&[], 4).is_err());
        assert!(unpack_i2s(&[0xFF], 0).is_err());
    }

    #[test]
    fn unpack_too_short() {
        let err = unpack_i2s(&[0x00], 8).unwrap_err();
        assert!(matches!(err, I2SError::PackedLengthMismatch { .. }));
    }

    #[test]
    fn pack_clamping_out_of_range() {
        let values: Vec<i8> = vec![-5, 5, -1, 0, 1, 100, -100, 0];
        let packed = pack_i2s(&values).unwrap();
        let unpacked = unpack_i2s(&packed, values.len()).unwrap();
        assert_eq!(unpacked, vec![-1, 1, -1, 0, 1, 1, -1, 0]);
    }

    #[test]
    fn roundtrip_32_elements() {
        let values: Vec<i8> = (0..32).map(|i| [-1, 0, 1][i % 3]).collect();
        let packed = pack_i2s(&values).unwrap();
        assert_eq!(packed.len(), 8);
        let unpacked = unpack_i2s(&packed, 32).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn roundtrip_256_elements() {
        let values: Vec<i8> = (0..256).map(|i| [-1, 0, 1][i % 3]).collect();
        let packed = pack_i2s(&values).unwrap();
        assert_eq!(packed.len(), 64);
        let unpacked = unpack_i2s(&packed, 256).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn roundtrip_non_multiple_of_4() {
        let values: Vec<i8> = vec![1, 0, -1, 1, 0];
        let packed = pack_i2s(&values).unwrap();
        let unpacked = unpack_i2s(&packed, values.len()).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn pack_single_value() {
        let packed = pack_i2s(&[1]).unwrap();
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_i2s(&packed, 1).unwrap();
        assert_eq!(unpacked, vec![1]);
    }

    #[test]
    fn roundtrip_large() {
        let values: Vec<i8> = (0..1024).map(|i| [-1, 0, 1][i % 3]).collect();
        let packed = pack_i2s(&values).unwrap();
        assert_eq!(packed.len(), 256);
        let unpacked = unpack_i2s(&packed, 1024).unwrap();
        assert_eq!(unpacked, values);
    }
}
